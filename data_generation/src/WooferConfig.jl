@with_kw struct WooferConfig
	MASS::Float64 = 8

	# Robot joint limits
	MAX_JOINT_TORQUE::Float64 = 12
	MAX_LEG_FORCE::Float64 = 133
	REVOLUTE_RANGE::Float64 = 3
	PRISMATIC_RANGE::Float64 = 0.18

	# Robot geometry
	LEG_FB::Float64 = 0.23			# front-back distance from center line to leg axis
	LEG_LR::Float64 = 0.104 		# left-right distance from center line to leg plane
	LEG_L::Float64  = 0.32
	ABDUCTION_OFFSET::Float64 = .074175 # distance from abduction axis to leg
	FOOT_RADIUS::Float64 = 0.02

	L::Float64 = 0.66
	W::Float64 = 0.176
	T::Float64 = 0.092
	Ix::Float64 = MASS/12 * (W^2 + T^2)
	Iy::Float64 = MASS/12 * (L^2 + T^2)
	Iz::Float64 = MASS/12 * (L^2 + W^2)
	INERTIA::Diagonal{Float64} = Diagonal([Ix, Iy, Iz])
	l0::Float64 = 0.18
	l1::Float64 = 0.32
	INV_INERTIA::Diagonal{Float64} = inv(INERTIA)
end

WOOFER_CONFIG = WooferConfig()
