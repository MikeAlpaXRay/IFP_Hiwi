import cv2 as cv
import numpy as np
import datetime
import os
import sys
from csv import writer
from parameters import \
    model_path, scene_path, result_path, scene_compare_path

def handleParameterArray(parameter_info):
    if parameter_info[3] == "s":
        array = list(np.arange(parameter_info[0] ,parameter_info[1], parameter_info[2]))
    elif parameter_info[3] == "c":
        array = list(np.linspace(parameter_info[0] ,parameter_info[1], parameter_info[2]))
    return array

def matching(python_parameters=[]):
    #################################################################
    # External Parameter input if needed
    #################################################################
    bat_parameter = sys.argv[1:]
    if len(bat_parameter) == 5:
        print("Parameters from .bat are used")
        relativeSamplingStep = float(parameter[0][:-1])
        relativeDistanceStep = float(parameter[1][:-1])
        numAngles = float(parameter[2][:-1])
        relativeSceneSampleStep = float(parameter[3][:-1])
        relativeSceneDistance = float(parameter[4])
    elif len(python_parameters) == 6:
        print("Parameters from funtion are used")
        counter = python_parameters[0]
        relativeSamplingStep = float(python_parameters[1])
        relativeDistanceStep = float(python_parameters[2])
        numAngles = float(python_parameters[3])
        relativeSceneSampleStep = float(python_parameters[4])
        relativeSceneDistance = float(python_parameters[5])
    else:
        print("Parameters from parameters.py are used")
        from parameters import \
            relativeSamplingStep, relativeDistanceStep, numAngles, \
            relativeSceneSampleStep, relativeSceneDistance



    #################################################################
    # get Model .PLY Files from /Model with prefix (ACTIVE_)
    #################################################################
    models = []
    possible_model_files = os.listdir(model_path)
    for possible_model in possible_model_files:
        if possible_model.endswith(".ply") and\
                (possible_model.startswith("ACTIVE_") or possible_model.startswith("AKTIVE_")):
            models.append(possible_model)


    #################################################################
    # get Scene .PLY Files from /Scene with prefix (ACTIVE_)
    #################################################################
    possible_scene_files = os.listdir(scene_path)
    for possible_scene in possible_scene_files:
        if possible_scene.endswith(".ply") and\
                (possible_scene.startswith("ACTIVE_") or possible_scene.startswith("AKTIVE_")):
            scene = possible_scene

    # Number of Results used
    N = 50


    if numAngles == 0:
        detector = cv.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep)
    else:
        detector = cv.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep, numAngles)

    for model in models:
        pc = cv.ppf_match_3d.loadPLYSimple(model_path + "/%s" % model, 1)


        tick1 = cv.getTickCount()
        detector.trainModel(pc)
        tick2 = cv.getTickCount()
        training_duration = (tick2 - tick1) / cv.getTickFrequency()
        print("Training complete in " + str(training_duration) + "sec")

        pcTest = cv.ppf_match_3d.loadPLYSimple(scene_path + "/%s" % scene, 1)

        tick1 = cv.getTickCount()
        results = detector.match(pcTest, relativeSceneSampleStep, relativeSceneDistance)
        tick2 = cv.getTickCount()
        matching_duration = (tick2 - tick1) / cv.getTickFrequency()
        print("Matching complete in " + str(matching_duration) + "sec")

        icp = cv.ppf_match_3d_ICP(100)
        _, results = icp.registerModelToScene(pc, pcTest, results[:N])

        now = datetime.datetime.now()
        print("####################### Current date and time: " + now.strftime("%Y %B%d - %H:%M:%S") +
              " #######################")


        # sort by resudial
        results.sort(key=lambda x: x.residual)

        # check if comparable pose exist
        scene_name = scene[7:-4]
        scene_pose_name = scene_name + "_pose.txt"

        possible_scene_pose_files = os.listdir(scene_compare_path)
        if scene_pose_name in possible_scene_pose_files:
            scene_pose = scene_compare_path + "/" + scene_pose_name

            pose = np.loadtxt(scene_pose, comments="#", delimiter=" ", unpack=False)
            scene_rotation = pose[:-1,:-1]
            scene_rotation_y = np.rad2deg(-np.arcsin(scene_rotation[2,0]))
            scene_rotation_x = np.rad2deg(np.arcsin(scene_rotation[2,1]))/np.cos(np.deg2rad(scene_rotation_y))
            scene_rotation_z = np.rad2deg(np.arcsin(scene_rotation[1,0]))/np.cos(np.deg2rad(scene_rotation_y))
            scene_translation = pose[:, -1][:-1]
            scene_translation_x = scene_translation[0]
            scene_translation_y = scene_translation[1]
            scene_translation_z = scene_translation[2]

        try:
            result = results[0]

            model_rotation = result.pose[:-1, :-1]
            model_rotation_y = np.rad2deg(-np.arcsin(model_rotation[2, 0]))
            model_rotation_x = np.rad2deg(np.arcsin(model_rotation[2, 1])) / np.cos(np.deg2rad(model_rotation_y))
            model_rotation_z = np.rad2deg(np.arcsin(model_rotation[1, 0])) / np.cos(np.deg2rad(model_rotation_y))
            model_translation = result.pose[:, -1][:-1]
            model_translation_x = model_translation[0]
            model_translation_y = model_translation[1]
            model_translation_z = model_translation[2]

            print("\n####################### Pose difference #######################")
            x_ax_dif = round(scene_rotation_x - model_rotation_x,5)
            y_ax_dif = round(scene_rotation_y - model_rotation_y,5)
            z_ax_dif = round(scene_rotation_z - model_rotation_z,5)
            x_dif = round(model_translation_x - scene_translation_x,5)
            y_dif = round(model_translation_y - scene_translation_y,5)
            z_dif = round(model_translation_z - scene_translation_z,5)


            print("around x: " + str(x_ax_dif) + "°")
            print("around y: " + str(y_ax_dif) + "°")
            print("around z: " + str(z_ax_dif) + "°")
            print("in x: " + str(x_dif))
            print("in x: " + str(y_dif))
            print("in x: " + str(z_dif))

            rot_norm = round(np.linalg.norm([x_ax_dif, y_ax_dif, z_ax_dif]), 5)
            tra_norm = round(np.linalg.norm([x_dif, y_dif, z_dif]), 5)

            print("\nRotation Error: " + str(rot_norm))
            print("Transtation Error: " + str(tra_norm))

            print("\n####################### Parameter documentation #######################")
            print("Rotation Error:\t\t\t\t" + str(rot_norm))
            print("Transtation Error:\t\t\t" + str(tra_norm))
            print("Training Duration:\t\t\t" + str(training_duration))
            print("Matching Duration:\t\t\t" + str(matching_duration))
            print("relativeSamplingStep:\t\t" + str(relativeSamplingStep))
            print("relativeDistanceStep:\t\t" + str(relativeDistanceStep))
            if not numAngles == 0:
                print("numAngles:\t\t\t\t\t" + str(numAngles))
            else:
                print("numAngles:\t\t\t\t\t-")
            print("relativeSceneSampleStep:\t" + str(relativeSceneSampleStep))
            print("relativeSceneDistance:\t\t" + str(relativeSceneDistance))
            print("Model:\t\t\t\t\t\t" + model)
            print("Scene:\t\t\t\t\t\t" + scene)
            print("Scene pose:\t\t\t\t\t" + scene_pose_name)
        except:
            rot_norm = "Error"
            tra_norm = "Error"

        # add Parameter to Outputfile
        paramter_doc = [counter, rot_norm, tra_norm,\
                       training_duration, matching_duration,\
                       relativeSamplingStep, relativeDistanceStep, numAngles,\
                       relativeSceneSampleStep, relativeSceneDistance,\
                       model, scene, scene_pose_name]
        with open('Parameterstudy.csv', 'a', newline='') as f_object:
            writer(f_object).writerow(paramter_doc)
            f_object.close()




def main():
    counter = 1
    #create Outputfile
    headersCSV = ["ID", "Rotation Error", "Transtation Error",\
                  "Training Duration [s]", "Matching Duration [s]",\
                  "relativeSamplingStep", "relativeDistanceStep", "numAngles",\
                  "relativeSceneSampleSte", "relativeSceneDistance",\
                  "Model", "Scene", "Scene pose"]
    with open('Parameterstudy.csv', 'a', newline='') as f_object:
        writer(f_object).writerow(headersCSV)
        f_object.close()

    # set Parameters
    # = [min, max, step/stepcount, "s"/"c"]
    relativeSamplingStep_Range = [0.01, 0.5, 3, "c"]
    relativeDistanceStep_Range = [0.01, 0.5, 1, "c"]
    numAngles_Range = [0, 50, 1, "s"]
    relativeSceneSampleSte_Range = [0.01, 0.5, 1, "c"]
    relativeSceneDistance_Range = [0.01, 0.5, 2, "c"]

    # transformParameters
    relativeSamplingStep_Range = handleParameterArray(relativeSamplingStep_Range)
    relativeDistanceStep_Range = handleParameterArray(relativeDistanceStep_Range)
    numAngles_Range = handleParameterArray(numAngles_Range)
    print(numAngles_Range)
    relativeSceneSampleSte_Range = handleParameterArray(relativeSceneSampleSte_Range)
    relativeSceneDistance_Range = handleParameterArray(relativeSceneDistance_Range)


    calulations = len(relativeSamplingStep_Range) * len(relativeDistanceStep_Range) * \
                  len(numAngles_Range) * len(relativeSceneSampleSte_Range) * len(relativeSceneDistance_Range)

    # Calc itter
    for i_relativeSamplingStep in relativeSamplingStep_Range:
        for i_relativeDistanceStep in relativeDistanceStep_Range:
            for i_numAngles in numAngles_Range:
                for i_relativeSceneSampleSte in relativeSceneSampleSte_Range:
                    for i_relativeSceneDistance in relativeSceneDistance_Range:
                        progress = round(counter / (calulations / 100), 2)
                        print(str(progress) + "%/100%")
                        parameter = [counter, i_relativeSamplingStep, i_relativeDistanceStep, i_numAngles, i_relativeSceneSampleSte, i_relativeSceneDistance]
                        matching(parameter)
                        counter += 1

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.rename('Parameterstudy.csv', '%s___Parameterstudy.csv' % (timestamp))





if __name__=="__main__":
    main()