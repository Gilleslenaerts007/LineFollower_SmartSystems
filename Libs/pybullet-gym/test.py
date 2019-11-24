from gym_line_follower.envs import LineFollowerEnv
env = LineFollowerEnv(gui=False, nb_cam_pts=8, max_track_err=0.3, speed_limit=0.2,
                      max_time=100, randomize=True, obsv_type="latch")