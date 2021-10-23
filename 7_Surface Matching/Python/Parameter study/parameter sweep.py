import cv2
import numpy as np
import os
import datetime
now = datetime.datetime.now()



#choosen objects for parameter study
scene_path = "given result\\scene001.ply"
obj0_path = "given result\\Mario000.ply"
obj1_path = "given result\\PeterRabbit015.ply"
obj2_path = "given result\Squirrel011.ply"
object_paths = [obj0_path, obj1_path, obj2_path]


# scene_path = "given result\\stanford\\Scene0View0_0.1.ply"
# obj0_path = "given result\\stanford\\Armadillo_vres2_small_scaled_0.ply"
# object_paths = [obj0_path]




#Algorithm Parameter
#Link: https://docs.opencv.org/4.5.3/db/d25/classcv_1_1ppf__match__3d_1_1PPF3DDetector.html#abe2433c0b4eb9be6506172aeccdd534e
#Detector
relativeSamplingStep = 0.04
relativeDistanceStep = 0.04
numAngles = 1
#Match
relativeSceneSampleStep = 0.04
relativeSceneDistance = 0.04
#ICP
#Link: https://docs.opencv.org/4.5.3/dc/d9b/classcv_1_1ppf__match__3d_1_1ICP.html
iterations = 500
tolerence = 0.005
rejectionScale = 2
numLevels = 4


##result folder name based of parameters
target_dir = str(relativeSamplingStep) + "_" + str(relativeDistanceStep) + "_" + str(numAngles) + "_" + str(relativeSceneSampleStep) + "_" +\
             str(relativeSceneDistance) + "_" + str(iterations) + "_" + str(tolerence) + "_" + str(rejectionScale) + "_" + str(numLevels)


#Create Result Dir
try:
    os.mkdir(target_dir)
    os.mkdir(target_dir + "\\objects")
except:
    print("Dir exists")


#nur test mit einem object
object_paths = [object_paths[0]]

#iterration über alle objekttypen
for obj in object_paths:
    info_string = "#######################Current date and time: " + now.strftime(
        '%H:%M:%S') + "#######################"
    print("#######################Current date and time: " + now.strftime('%H:%M:%S') + "#######################")
    pc_path = obj
    pc = cv2.ppf_match_3d.loadPLYSimple(pc_path)

    tick1 = cv2.getTickCount()
    info_string += "\n" + "Start training... on " + obj.split("\\")[1]
    print("Start training... on " + obj.split("\\")[1])
    #detector = cv2.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep)
    detector = cv2.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep, numAngles)
    try:
        detector.trainModel(pc)
    except:
        break
    tick2 = cv2.getTickCount()
    info_string += "\n" + "Training complete in " + str((tick2 - tick1) / cv2.getTickFrequency()) + "sec"
    print("Training complete in " + str((tick2-tick1)/ cv2.getTickFrequency()) + "sec")
    info_string += "\n" + "Loading model..."
    print("Loading model...")
    pcTest_path = scene_path
    pcTest = cv2.ppf_match_3d.loadPLYSimple(pcTest_path)

    tick1 = cv2.getTickCount()
    info_string += "\nStarting matching... with " + scene_path.split("\\")[1]
    print("Starting matching... with " + scene_path.split("\\")[1])
    try:
        results = detector.match(pcTest, relativeSceneSampleStep, relativeSceneDistance)
    except:
        break
    tick2 = cv2.getTickCount()

    info_string += "\nPPF Elapsed Time " + str((tick2 - tick1) / cv2.getTickFrequency()) + "sec"
    print("PPF Elapsed Time " + str((tick2 - tick1) / cv2.getTickFrequency()) + "sec")
    N = 2
    resultsSub = results[:N]
    icp = cv2.ppf_match_3d_ICP(iterations, tolerence, rejectionScale, numLevels)
    # icp = cv2.ppf_match_3d_ICP()
    t1 = cv2.getTickCount()

    info_string += "\nPerforming ICP on " + str(N) + " poses..."
    print("Performing ICP on " + str(N) + " poses...")
    icp.registerModelToScene(pc, pcTest)
    t2 = cv2.getTickCount()

    info_string += "\nICP Elapsed Time " + str((tick2 - tick1) / cv2.getTickFrequency()) + "sec"
    print("ICP Elapsed Time " + str((tick2 - tick1) / cv2.getTickFrequency()) + "sec")

    info_string += "\nPoses: "
    print("Poses: ")

    i = 0
    while i < len(resultsSub):
        result = resultsSub[i]
        info_string += "\nPose Result " + str(i) + "\n"
        print("Pose Result " + str(i))
        info_string += str(result.pose)
        cv2.ppf_match_3d_Pose3D.printPose(result)

        #if i == 0:
        pct = cv2.ppf_match_3d.transformPCPose(pc, result.pose)
        cv2.ppf_match_3d.writePLY(pct, target_dir + "\\objects\\" + str(
            pc_path.split("\\")[1][:-4]) + "_" + str(pcTest_path.split("\\")[1][:-4]) + "_" + str(i) + ".ply")
        i += 1
    f = open(target_dir + "\\" + str(pc_path.split("\\")[1][:-4]) + "_" + str(pcTest_path.split("\\")[1][:-4]) + "_poses.txt",
             'w')
    f.write(info_string)
    f.close()