# Used only by El-Sewisy 2020

class SimResult:
    def __init__(self, label, times, avs, realDirections, decDirections, errs_signed, errs, quad_thetas, quad_dirs, quad_errs_signed, quad_errs):
        self.label = label
        # timestamps in seconds starting from 0
        self.times = times
        # angular velocities per timestep
        self.avs = avs
        # ground truth directions per timestep
        self.realDirections = realDirections
        # decoded directions per timestep
        self.decDirections = decDirections
        # signed errors HDC network vs. ground truth per timestep
        self.errs_signed = errs_signed
        # unsigned errors HDC network vs. ground truth per timestep
        self.errs = errs
        # change in angle using using (angular velocity * dt) every timestep
        self.quad_thetas = quad_thetas
        # directions obtained by summing (angular velocity * dt) every timestep
        self.quad_dirs = quad_dirs
        # signed error ground truth vs. sumDirs every timestep
        self.quad_errs_signed = quad_errs_signed
        # unsigned error ground truth vs. sumDirs every timestep
        self.quad_errs = quad_errs