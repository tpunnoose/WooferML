@with_kw struct RandomParams
	next_foot_loc::Vector{Float64} = zeros(12)

	wn_cart::Float64 = 20
	zeta_cart::Float64 = 0.8
	kp_cart::Float64 = wn_cart^2
	kd_cart::Float64 = 2*wn_cart*zeta_cart
	J::Matrix{Float64} = zeros(3,3)
end

function calcTorques!(torques::Vector{T}, cur_pos::Vector{T}, cur_vel::Vector{T}, pos_des::Vector{T}, vel_des::Vector{T}, α::Vector{T}, params::RandomParams) where {T<:Number}
	#=
	PD cartesian controller around swing leg trajectory
	=#

	legJacobian!(params.J, α)

	torques .= params.J' * (params.kp_cart*(pos_des - cur_pos) + params.kd_cart*(vel_des - cur_vel))
end
