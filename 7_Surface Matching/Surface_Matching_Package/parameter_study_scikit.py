import cv2 as cv
import numpy as np
import datetime
import os
import sys
from csv import writer
from parameters import \
    model_path, scene_path, result_path, scene_compare_path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import logging


def getModels(path):
    #################################################################
    # get Model .PLY Files from /Model with prefix (ACTIVE_)
    #################################################################
    models = []
    possible_model_files = os.listdir(path)
    for possible_model in possible_model_files:
        if possible_model.endswith(".ply") and \
                (possible_model.startswith("ACTIVE_") or possible_model.startswith("AKTIVE_")):
            models.append(possible_model)
    return models


def getScenes(path):
    #################################################################
    # get Scene .PLY Files from /Scene with prefix (ACTIVE_)
    #################################################################
    scenes = []
    possible_scene_files = os.listdir(path)
    for possible_scene in possible_scene_files:
        if possible_scene.endswith(".ply") and \
                (possible_scene.startswith("ACTIVE_") or possible_scene.startswith("AKTIVE_")):
            scenes.append(possible_scene)
    return scenes

def matching(python_parameters=[]):
    #################################################################
    # External Parameter input
    #################################################################
    relativeSamplingStep = float(python_parameters[0])
    relativeDistanceStep = float(python_parameters[1])
    numAngles = float(python_parameters[2])
    relativeSceneSampleStep = float(python_parameters[3])
    relativeSceneDistance = float(python_parameters[4])


    #################################################################
    # get Model .PLY Files from /Model with prefix (ACTIVE_)
    #################################################################
    
    #models = getModels(model_path)
    
    
    ### small hack reverse
    models = []
    models.append("ACTIVE_model_palette_n_100000_1.ply")

    #################################################################
    # get Scene .PLY Files from /Scene with prefix (ACTIVE_)
    #################################################################
    
    #scenes = getScenes(scene_path)
    
    
    ### small hack reverse
    scenes = []
    scenes.append("ACTIVE_scene_palette_n_10000.ply")

    # Number of Results used
    N = 50
    if numAngles == 0:
        detector = cv.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep)
    else:
        detector = cv.ppf_match_3d_PPF3DDetector(relativeSamplingStep, relativeDistanceStep, numAngles)
    for model in models:
        logging.info(f"\n\nTraining with model: {model}")
        for scene in scenes:
            logging.info(f"Training with scene: {scene}")
            logging.info(f"With Parameter: {python_parameters}")
            tick1 = cv.getTickCount()
            pc = cv.ppf_match_3d.loadPLYSimple(model_path + "/%s" % model, 1)
            tick2 = cv.getTickCount()
            modal_load_duration = (tick2 - tick1) / cv.getTickFrequency()

            #print("Modelloading complete in " + str(modal_load_duration) + "sec")

            tick1 = cv.getTickCount()
            detector.trainModel(pc)
            tick2 = cv.getTickCount()
            training_duration = (tick2 - tick1) / cv.getTickFrequency()

            #print("Training complete in " + str(training_duration) + "sec")

            tick1 = cv.getTickCount()
            pcTest = cv.ppf_match_3d.loadPLYSimple(scene_path + "/%s" % scene, 1)
            tick2 = cv.getTickCount()
            scene_load_duration = (tick2 - tick1) / cv.getTickFrequency()

            #print("Sceneloading complete in " + str(scene_load_duration) + "sec")

            tick1 = cv.getTickCount()
            results = detector.match(pcTest, relativeSceneSampleStep, relativeSceneDistance)
            tick2 = cv.getTickCount()
            matching_duration = (tick2 - tick1) / cv.getTickFrequency()

            #print("Matching complete in " + str(matching_duration) + "sec")

            #times = [modal_load_duration, training_duration, scene_load_duration, matching_duration]

            icp = cv.ppf_match_3d_ICP(100)
            _, results = icp.registerModelToScene(pc, pcTest, results[:N])

            now = datetime.datetime.now()

            #print("####################### Current date and time: " + now.strftime("%Y %B%d - %H:%M:%S") +
            #      " #######################")



            # sort by resudial
            list(results).sort(key=lambda x: x.residual)

            # check if comparable pose exist
            scene_name = scene[7:-4]
            scene_pose_name = scene_name + "_pose.txt"

            possible_scene_pose_files = os.listdir(scene_compare_path)
            if scene_pose_name in possible_scene_pose_files:
                scene_pose = scene_compare_path + "/" + scene_pose_name

                pose = np.loadtxt(scene_pose, comments="#", delimiter=" ", unpack=False)
                scene_rotation = pose[:-1, :-1]
                scene_rotation_y = np.rad2deg(-np.arcsin(scene_rotation[2, 0]))
                scene_rotation_x = np.rad2deg(np.arcsin(scene_rotation[2, 1])) / np.cos(np.deg2rad(scene_rotation_y))
                scene_rotation_z = np.rad2deg(np.arcsin(scene_rotation[1, 0])) / np.cos(np.deg2rad(scene_rotation_y))
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

                x_ax_dif = round(scene_rotation_x - model_rotation_x, 5)
                y_ax_dif = round(scene_rotation_y - model_rotation_y, 5)
                z_ax_dif = round(scene_rotation_z - model_rotation_z, 5)
                x_dif = round(model_translation_x - scene_translation_x, 5)
                y_dif = round(model_translation_y - scene_translation_y, 5)
                z_dif = round(model_translation_z - scene_translation_z, 5)


                rot_norm = round(np.linalg.norm([x_ax_dif, y_ax_dif, z_ax_dif]), 5)
                tra_norm = round(np.linalg.norm([x_dif, y_dif, z_dif]), 5)

                # starker einfluss von translation geringerer von rotation
                calc_score = 10*(tra_norm**4) + (rot_norm**2)/10
                print("Rotation Error: " + str(rot_norm) + "\tTranstation Error: " + str(tra_norm) +"\tScore: " + str(calc_score))

                logging.info("Rotation Error: " + str(rot_norm) + "\tTranstation Error: " + str(tra_norm) +"\tScore: " + str(calc_score))
            except:
                rot_norm = "Error"
                tra_norm = "Error"
                calc_score = 99999999999999

            return calc_score



def main():

    logging.basicConfig(filename='quick.log', encoding='utf-8', level=logging.INFO)

    space = [Real(0.01, 0.1, name='relativeSamplingStep_Range'),
             Real(0.025, 0.1, name='relativeDistanceStep_Range'),
             Integer(0, 25, name='numAngles_Range'),
             Real(0.1, 1, name='relativeSceneSampleSte_Range'),
             Real(0.01, 0.1, name='relativeSceneDistance_Range')]



    res_gp = gp_minimize(matching, space, n_calls=15, random_state=0)

    print("Best score=%.4f" % res_gp.fun)

    print("""Best parameters:
    - max_depth=%f
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%f
    - min_samples_leaf=%f""" % (res_gp.x[0], res_gp.x[1],
                                res_gp.x[2], res_gp.x[3],
                                res_gp.x[4]))

    plot_convergence(res_gp)




if __name__ == "__main__":
    main()
