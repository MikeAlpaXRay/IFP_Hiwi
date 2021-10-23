import cv2
import numpy as np
import os
import datetime
now = datetime.datetime.now()


#In welchen Szenen ist das Object zu finden, siehe Content.xlsx
doll = ["05", "06", "10", "13", "22", "30", "39"]
duck = ["11", "13","14"]
frog = ["18", "19", "22", "30", "31", "32", "39", "40"]
mario = ["01", "03", "05", "06", "10", "14", "19", "31", "32", "36", "40"]
peterrabbit = ["01", "03", "05", "11", "18", "19", "22", "32", "36", "39", "40"]
squirrel = ["01", "03", "06", "13", "14", "30", "31", "32", "36"]



#Algorithm Parameter
#Link: https://docs.opencv.org/4.5.3/db/d25/classcv_1_1ppf__match__3d_1_1PPF3DDetector.html#abe2433c0b4eb9be6506172aeccdd534e
#Detector
relativeSamplingStep = 0.1
relativeDistanceStep = 0.05
#numAngles = 1
#Match
relativeSceneSampleStep = 0.5
relativeSceneDistance = 0.05
#ICP
#Link: https://docs.opencv.org/4.5.3/dc/d9b/classcv_1_1ppf__match__3d_1_1ICP.html
iterations = 100
tolerence = 0.005
rejectionScale = 2.5
numLevels = 8

# target_dir = str(relativeSamplingStep) + "_" + str(relativeDistanceStep) + "_" + str(numAngles) + "_" + str(relativeSceneSampleStep) + "_" +\
#              str(relativeSceneDistance) + "_" + str(iterations) + "_" + str(tolerence) + "_" + str(rejectionScale) + "_" + str(numLevels)
target_dir = str(relativeSamplingStep) + "_" + str(relativeDistanceStep) + "_" + str(relativeSceneSampleStep) + "_" +\
             str(relativeSceneDistance) + "_" + str(iterations) + "_" + str(tolerence) + "_" + str(rejectionScale) + "_" + str(numLevels)



#Path to objectfolders
obj_folders = os.listdir("Kinect_readable\MeshRegistration")
#Path to scenes
sce_files = os.listdir("Kinect_readable\ObjectRecognition")[1:]

#Create Result Dir
try:
    os.mkdir("Full run\\result\\" + target_dir)
    os.mkdir("Full run\\result\\" + target_dir + "\\objects")
except:
    print("Dir exists")


#iterration über alle objekttypen
for obj_folder in obj_folders:
    try:
        #Finde alle schon berechneten Kombinationen
        calc_sce = [item for item in os.listdir("Full run\\result\\" + target_dir) if "poses" and obj_folder in item]
    except:
        calc_sce = []
    try:
        #Welche verschiedenen Versionen gibt es je objekttyp
        obj_files = os.listdir("Kinect_readable\MeshRegistration\\" + str(obj_folder))
    except:
        break
    #get scene_numbers for specific object
    for obj_file in obj_files:
        if "Doll" in obj_file:
            scenes_no = doll
        elif "duck" in obj_file:
            scenes_no = duck
        elif "Frog" in obj_file:
            scenes_no = frog
        elif "mario" in obj_file:
            scenes_no = mario
        elif "PeterRabbit" in obj_file:
            scenes_no = peterrabbit
        elif "Squirrel" in obj_file:
            scenes_no = squirrel
        #welche scenen wurden schon berechnet
        calc_obj = [item[:-10][-2:] for item in calc_sce if obj_file[:-4] in item]
        if len(calc_obj) < len(scenes_no):
            info_string = "#######################Current date and time: " + now.strftime('%H:%M:%S') + "#######################"
            print("#######################Current date and time: " + now.strftime('%H:%M:%S') + "#######################")
            pc_path = "Kinect_readable/MeshRegistration/" + str(obj_folder) + "/" + str(obj_file)
            pc = cv2.ppf_match_3d.loadPLYSimple(pc_path)

            tick1 = cv2.getTickCount()
            info_string += "\n" + "Start training... on " + str(obj_file[:-4])
            print("Start training... on " + str(obj_file[:-4]))
            detector = cv2.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep)
            #detector = cv2.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep, numAngles)
            try:
                detector.trainModel(pc)
            except:
                break
            tick2 = cv2.getTickCount()
            info_string += "\n" + "Training complete in " + str((tick2-tick1)/ cv2.getTickFrequency()) + "sec"
            print("Training complete in " + str((tick2-tick1)/ cv2.getTickFrequency()) + "sec")
            info_string += "\n" + "Loading model..."
            print("Loading model...")

            for sce_file in sce_files:
                #nur scenen werden gematched die gebraucht sind und noch nicht berechnet wurden
                if sce_file[:-4][-2:] in scenes_no and sce_file[:-4][-2:] not in calc_obj:
                    f = open("Full run\\result\\" + target_dir + "\\" + str(obj_file[:-4]) + "_" + str(sce_file[:-4]) + "_poses.txt", 'w')
                    f.write(info_string)
                    pcTest_path = "Kinect_readable/ObjectRecognition/" + str(sce_file)
                    pcTest = cv2.ppf_match_3d.loadPLYSimple(pcTest_path)

                    tick1 = cv2.getTickCount()
                    f.write("\nStarting matching... with " + str(sce_file[:-4]))
                    print("Starting matching... with " + str(sce_file[:-4]))
                    try:
                        results = detector.match(pcTest, relativeSceneSampleStep, relativeSceneDistance)
                    except:
                        break
                    tick2 = cv2.getTickCount()
                    f.write("\nPPF Elapsed Time " + str((tick2-tick1)/ cv2.getTickFrequency()) + "sec")
                    print("PPF Elapsed Time " + str((tick2-tick1)/ cv2.getTickFrequency()) + "sec")

                    N = 2
                    resultsSub = results[:N]
                    icp = cv2.ppf_match_3d_ICP(iterations, tolerence, rejectionScale, numLevels)
                    #icp = cv2.ppf_match_3d_ICP()
                    t1 = cv2.getTickCount()

                    f.write("\nPerforming ICP on " + str(N) + " poses...")
                    print("Performing ICP on " + str(N) + " poses...")
                    icp.registerModelToScene(pc, pcTest)
                    t2 = cv2.getTickCount()

                    f.write("\nICP Elapsed Time " + str((tick2-tick1)/cv2.getTickFrequency()) + "sec")
                    print("ICP Elapsed Time " + str((tick2-tick1)/cv2.getTickFrequency()) + "sec")

                    f.write("\nPoses: ")
                    print("Poses: ")

                    i = 0
                    while i < len(resultsSub):
                        result = resultsSub[i]
                        f.write("\nPose Result " + str(i) + "\n")
                        print("Pose Result " + str(i))
                        f.write(str(result.pose))
                        cv2.ppf_match_3d_Pose3D.printPose(result)

                        # if i == 0:
                        pct = cv2.ppf_match_3d.transformPCPose(pc, result.pose)
                        cv2.ppf_match_3d.writePLY(pct, "Full run\\result\\" + target_dir + "\\objects\\" + str(obj_file[:-4]) + "_" + str(sce_file[:-4]) + "_" + str(i) + ".ply")
                        i += 1
                    f.close()
