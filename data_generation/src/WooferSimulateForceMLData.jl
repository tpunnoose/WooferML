function simulate()
   # look at local directory
   pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)

   # parse MuJoCo XML file
   xmlparser = pyimport("WooferXMLParser")
   xmlparser.Parse()

   s = loadmodel("src/woofer_fixed_base_out.xml", 1200, 900)

   d = s.d
   m = s.m

   lower_dt = 0.001
   upper_dt = 0.75
   data_dt = 0.001
   x = zeros(13)
   accel = zeros(3)
   gyro = zeros(3)
   joint_pos = zeros(12)
   joint_vel = zeros(12)
   torques = zeros(12)
   torque_i = zeros(3)

   J = zeros(3,3)
   params = RandomParams()
   cur_pos = zeros(3)
   cur_vel = zeros(3)

   pos_des = [0, 0, -0.265]
   # pos_des = [0.0, 0.0, -0.4]
   vel_des = zeros(3)

   # df = DataFrame(q1=Float64[], q2=Float64[], q3=Float64[], q1_dot=Float64[], q2_dot=Float64[], q3_dot=Float64[], torque_x=Float64[], torque_y=Float64[], torque_z=Float64[])
   data = Array{Float64}[]

   # Loop until the user closes the window
   WooferSim.alignscale(s)
   while !GLFW.WindowShouldClose(s.window)
      ### basically sim step so things don't have to be defined multiple times
      if s.paused
         if s.pert[].active > 0
            mjv_applyPerturbPose(m, d, s.pert, 1)  # move mocap and dynamic bodies
            mj_forward(m, d)
         end
      else
         #slow motion factor: 10x
         factor = s.slowmotion ? 10 : 1

         # advance effective simulation time by 1/refreshrate
         startsimtm = d.d[].time
         starttm = time()
         refreshtm = 1.0/(factor*s.refreshrate)
         updates = refreshtm / m.m[].opt.timestep

         steps = round(Int, round(s.framecount+updates)-s.framecount)
         s.framecount += updates

         for i=1:steps
            # clear old perturbations, apply new
            d.xfrc_applied .= 0.0

            if s.pert[].select > 0
               mjv_applyPerturbPose(m, d, s.pert, 0) # move mocap bodies only
               mjv_applyPerturbForce(m, d, s.pert)
            end

            t = d.d[].time

            if t % upper_dt < 1e-3
               δ_x = 0.3*(rand(Float64) - 0.5)
               δ_y = 0.15*(rand(Float64) - 0.5)
               δ_z = 0.14*(rand(Float64) - 0.5)
               pos_des = [0, 0, -0.37] + [δ_x, δ_y, δ_z]
            end

            # lower level update loop (eg state estimation, torque updates)
            if t % lower_dt < 1e-3
               # ground truth states
               x[1:3] .= s.d.qpos[1:3]
               x[4:7] .= s.d.qpos[4:7]
               x[8:10] .= s.d.qvel[1:3]
               x[11:13] .= s.d.qvel[4:6]

               accel       .= s.d.sensordata[1:3]
               gyro        .= s.d.sensordata[4:6]
               joint_pos   .= s.d.sensordata[7:18]
               joint_vel   .= s.d.sensordata[19:30]


               for i=1
                  α = joint_pos[3*(i-1)+1:3*(i-1)+3]
                  legJacobian!(J, α)
                  cur_vel = J*α
                  if i==2 || i==4
                     cur_pos .= forwardKinematicsFromHip(α, true)
                     calcTorques!(torque_i, cur_pos, cur_vel, pos_des + [0, WOOFER_CONFIG.ABDUCTION_OFFSET, 0], vel_des, α, params)
                  else
                     cur_pos .= forwardKinematicsFromHip(α, false)
                     calcTorques!(torque_i, cur_pos, cur_vel, pos_des + [0, -WOOFER_CONFIG.ABDUCTION_OFFSET, 0], vel_des, α, params)
                  end
                  torques[3*(i-1)+1:3*(i-1)+3] .= torque_i
               end

               s.d.ctrl .= torques
            end

            if t % data_dt < 1e-3
               push!(data, [joint_pos[1:3]..., joint_vel[1:3]..., torque_i...])
            end

            mj_step(s.m, s.d)

            # break on reset
            (d.d[].time < startsimtm) && break
         end
      end
      render(s, s.window)
      GLFW.PollEvents()
   end
   GLFW.DestroyWindow(s.window)

   n_rows = Int(floor(length(data)/3))

   formatted_data = zeros(n_rows, 21)

   for i=1:n_rows
      start_index = 3*(i-1)+1
      prev_joint_i = data[start_index][1:6]
      joint_i = data[start_index+1][1:6]
      torque_i = data[start_index+1][7:9]
      next_joint_i = data[start_index+2][1:6]

      formatted_data[i,:] = [prev_joint_i..., joint_i..., next_joint_i..., torque_i...]
   end

   open("force_test.csv", "w") do io
      writedlm(io, formatted_data, ',')
   end;
end
