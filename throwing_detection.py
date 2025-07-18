import mp_helper
import sys
import matplotlib.pyplot as plt
deliveries_import_filename = sys.argv[1]
filenames_filename = sys.argv[2]

filenames = mp_helper.import_filenames(filenames_filename)

landmarks = mp_helper.import_landmarks(deliveries_import_filename)

wp, ep, sp, rfs = mp_helper.get_release_points_of_dir(landmarks)

#video_1 = mp_helper.get_landmarks_by_video_index(1,landmarks)

#rf_v1 = mp_helper.get_landmarks_by_frame_index(rfs[1],video_1)

#mp_helper.plot_skeleton(rf_v1)

angles_by_video = mp_helper.calc_elbow_angle(wp,ep,sp)

fig, ax = plt.subplots()

for x in range(0,len(rfs)):
    print(f"Video {filenames[x]} release point {rfs[x]}")
print(f"Filenames length: {len(filenames)}")
print(f"Angles length: {len(angles_by_video)}")

for i in range(0,len(angles_by_video)):
    label = filenames[i]
    angle = angles_by_video[i]
    p = ax.bar(label, angle, 0.5,color="b")

ax.set_title("Angle of elbow flexion at release by video")
ax.set_xlabel("Delivery Number")
ax.set_ylabel("Elbow flexion (degrees)")
ax.tick_params(rotation=90)
plt.axhline(y=15,linewidth=1,color="r")
plt.show()
        
