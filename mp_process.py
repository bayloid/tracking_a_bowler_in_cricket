import mp_helper
import sys

video_dir = sys.argv[1]
export_file_name = sys.argv[2]
dir_landmarks = mp_helper.process_dir(video_dir,
                                      export=True,
                                      export_name= export_file_name)




