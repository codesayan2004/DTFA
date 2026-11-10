"""Dynamic Bridging Tuning (DBT).

Extension of DTFA's Adaptive Feature Bridging (utils/AFB/loss.py). The baseline aligns the student
with the IRT and SPT teachers through two independent bridges with a fixed 1:1 weighting:

    L = 0.2 * L_det + L_rc(F_S, F_IRT) + L_dc(F_S, F_SPT)

DBT instead fuses the two (bridged) teachers with a learned, spatially varying gate and aligns the
student with the fused target:

    F_B(x, y) = a(x, y) * F~_IRT(x, y) + (1 - a(x, y)) * F~_SPT(x, y)
    a(x, y)   = r_I / (r_I + r_S + eps)                 reliability-aware fusion, r >= 0
    [z_I, z_S] = gate([F~_IRT; F~_SPT; F_S; W; D])       student- and severity-aware gate
    r_I = softplus(z_I + beta * W),  r_S = softplus(z_S) (severity prior: beta > 0 -> more IRT in fog)
    W   = f_w(I_a) in [0, 1]^{H x W}                     learned weather-severity map
    D   = ||F~_IRT - F~_SPT||_2 / mean(.)                 teacher disagreement

    L = 0.2 * L_det + lambda * L_align(F_S, sg(F_B)) + eta * L_gate(sg(F_S), F_B)
        + gamma * L_reg(a) + mu * L_sev(W)

  L_align : trains the STUDENT: masked generative distillation towards the fused target (same
            operator as the baseline AFB, MgdLoss), with the gate output a detached.
  L_gate  : trains the GATE: ||F_S - F_B||^2 / ||F~_IRT - F~_SPT||^2 with the student detached
            (the plan's "distillation-based self-guidance"). The MGD generator is a fixed random
            conv, so L_align alone gives the gate no usable signal about which teacher the student
            is closer to; the direct, scale-normalised loss does.
  L_reg   : 1 - H(a) / log 2, discourages the gate from collapsing onto one teacher
  L_sev   : optional self-supervision of W with the paired clean image,
            W* = clip(mean_c |I_a - I_c| / sev_scale, 0, 1)   (training data is paired)

As in the baseline, the bridge projection and the MGD generator keep their random initialisation
(the original train_AWD.py never adds the AFB parameters to the optimizer); only the gate, the
severity estimator and beta are trained, with their own Adam optimizer (train_AWD.py, DBT_LR). The whole module is used during training only, so the
deployed detector is unchanged and has no extra cost.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.AFB.loss import MgdLoss, ShallowPym

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SeverityEstimator(nn.Module):
    """f_w: adverse image (normalised, 3 x H x W) -> severity logits at 1/4 resolution."""

    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.SiLU(),
            nn.Conv2d(16, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU(),
            nn.Conv2d(width, width, 3, 1, 2, dilation=2, bias=False), nn.BatchNorm2d(width), nn.SiLU(),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, img):
        return self.net(img)  # logits; sigmoid gives W in [0, 1]


def severity_target(img_a, img_c, size, scale):
    """Pseudo-label for local weather severity from the paired clean image."""
    mean = img_a.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = img_a.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    diff = ((img_a - img_c) * std).abs().mean(1, keepdim=True)  # |I_a - I_c| in [0, 1] pixel units
    diff = F.adaptive_avg_pool2d(diff, size)
    return (diff / scale).clamp(0, 1)


class DBTLoss(nn.Module):
    MODES = ("spatial", "image", "fixed")

    def __init__(self, alpha_mgd=1e-7, lambda_mgd=0.65, mode="spatial", use_student=True,
                 use_severity=True, use_disagreement=True, lam=2.0, eta=1.0, gamma=0.01, mu=0.1,
                 sev_scale=0.25, gate_width=64, channels=128, eps=1e-6):
        super().__init__()
        assert mode in self.MODES, f"DBT mode must be one of {self.MODES}"
        if mode == "fixed":  # ablation: plain average of the two bridged teachers, no gate at all
            use_student = use_severity = use_disagreement = False
        self.config = dict(alpha_mgd=alpha_mgd, lambda_mgd=lambda_mgd, mode=mode, use_student=use_student,
                           use_severity=use_severity, use_disagreement=use_disagreement, lam=lam, eta=eta,
                           gamma=gamma, mu=mu, sev_scale=sev_scale, gate_width=gate_width, channels=channels)
        self.mode, self.lam, self.eta, self.gamma, self.mu = mode, lam, eta, gamma, mu
        self.use_student, self.use_severity, self.use_disagreement = use_student, use_severity, use_disagreement
        self.sev_scale, self.eps = sev_scale, eps

        # Bridge + masked generative alignment: same operators as the baseline AFB.
        self.bridge = ShallowPym()
        self.align = MgdLoss(channels, channels, alpha_mgd, lambda_mgd)

        # Gating network (mixture of teacher features).
        in_ch = 2 * channels + (channels if use_student else 0) + int(use_severity) + int(use_disagreement)
        self.severity = SeverityEstimator() if use_severity else None
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, gate_width, 3, 1, 1, bias=False), nn.BatchNorm2d(gate_width), nn.SiLU(),
            nn.Conv2d(gate_width, 2, 1),
        )
        self.beta = nn.Parameter(torch.tensor(1.0))  # severity prior on the IRT logit
        self.reset_gate()

    def reset_gate(self):
        """Start from an even split between the teachers (a = 0.5 when W = 0)."""
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        with torch.no_grad():
            self.beta.fill_(1.0)

    def trainable_parameters(self):
        """Parameters the optimizer should update (bridge + generator stay fixed, as in the baseline)."""
        params = list(self.gate.parameters()) + [self.beta]
        if self.severity is not None:
            params += list(self.severity.parameters())
        return params

    def describe(self):
        parts = ["F_IRT", "F_SPT"] + (["F_S"] if self.use_student else []) + \
                (["W"] if self.use_severity else []) + (["D"] if self.use_disagreement else [])
        if self.mode == "fixed":
            return f"mode=fixed (alpha=0.5, no gate), lambda={self.lam}"
        return (f"mode={self.mode}, gate inputs=[{', '.join(parts)}], lambda={self.lam}, eta={self.eta}, "
                f"gamma={self.gamma}, mu={self.mu if self.use_severity else 0}")

    def forward(self, s_feats, irt_feats, spt_feats, img_a=None, img_c=None):
        """s_feats / irt_feats / spt_feats: (dark3 128x80x80, dark2 64x160x160) like the baseline AFB."""
        F_S = self.bridge(*s_feats)[1]
        F_I = self.bridge(*irt_feats)[1]
        F_P = self.bridge(*spt_feats)[1]
        N, _, H, W_ = F_S.shape

        sev_logits = sev = None
        if self.use_severity:
            sev_logits = self.severity(img_a)
            if sev_logits.shape[-2:] != (H, W_):
                sev_logits = F.interpolate(sev_logits, size=(H, W_), mode="bilinear", align_corners=False)
            sev = torch.sigmoid(sev_logits)

        disagreement = None
        if self.use_disagreement:
            d = (F_I - F_P).detach().pow(2).sum(1, keepdim=True).sqrt()
            disagreement = d / (d.mean(dim=(2, 3), keepdim=True) + self.eps)

        if self.mode == "fixed":
            alpha = F_S.new_full((N, 1, H, W_), 0.5)
        else:
            inputs = [F_I.detach(), F_P.detach()]
            if self.use_student:
                inputs.append(F_S.detach())  # student-aware, but the gate cannot push the student around
            if self.use_severity:
                inputs.append(sev)
            if self.use_disagreement:
                inputs.append(disagreement)
            z = self.gate(torch.cat(inputs, 1))
            z_i, z_s = z[:, :1], z[:, 1:]
            if self.use_severity:
                z_i = z_i + self.beta * sev
            if self.mode == "image":  # Option A: one weight per image
                z_i, z_s = z_i.mean(dim=(2, 3), keepdim=True), z_s.mean(dim=(2, 3), keepdim=True)
            r_i, r_s = F.softplus(z_i), F.softplus(z_s)
            alpha = r_i / (r_i + r_s + self.eps)
            alpha = alpha.expand(N, 1, H, W_)

        a_sg = alpha.detach()
        l_align = self.align(F_S, a_sg * F_I + (1 - a_sg) * F_P)  # student <- fused target

        if self.mode == "fixed":
            l_gate = l_reg = F_S.new_zeros(())
        else:
            fused = alpha * F_I.detach() + (1 - alpha) * F_P.detach()  # gate <- which mix fits the student
            l_gate = (F_S.detach() - fused).pow(2).mean() / ((F_I - F_P).detach().pow(2).mean() + self.eps)
            a = alpha.clamp(1e-6, 1 - 1e-6)
            entropy = -(a * a.log() + (1 - a) * (1 - a).log())
            l_reg = (1 - entropy / math.log(2)).mean()

        l_sev = F_S.new_zeros(())
        if self.use_severity and img_c is not None and self.mu > 0:
            target = severity_target(img_a, img_c, (H, W_), self.sev_scale)
            l_sev = F.binary_cross_entropy_with_logits(sev_logits, target)

        loss = self.lam * l_align + self.eta * l_gate + self.gamma * l_reg + self.mu * l_sev

        with torch.no_grad():
            a = alpha.float()
            stats = {"alpha_mean": a.mean().item(), "alpha_std": a.std().item()}
            if sev is not None:
                w = sev.float()
                stats["W_mean"] = w.mean().item()
                av, wv = a.flatten() - a.mean(), w.flatten() - w.mean()
                denom = av.norm() * wv.norm()
                stats["corr_alpha_W"] = (av @ wv / denom).item() if denom > 0 else 0.0
        return {"loss": loss, "align": l_align, "gate": l_gate, "reg": l_reg, "sev": l_sev,
                "alpha": alpha.detach(), "W": None if sev is None else sev.detach(),
                "D": None if disagreement is None else disagreement, "stats": stats}
