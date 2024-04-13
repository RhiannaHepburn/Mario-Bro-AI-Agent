# Authors: Brittany Carlsson (22752092), Rhianna Hepburn (23340238)  
# Agent 1: Rule-Based Heuristic 
# Unit: CITS3001 Algorithms, Agents and Artificail Intelligence (Semester Two 2023)
# Assignment: Super Mario Project


# List of Imports 
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gym
import numpy as np 
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from PIL import Image 

#This functions gathers information regarding the RGB values of Mario's immediate environment (10 pixels in front of the agent, and 5 pixels on the ground) 
def information(mario_coords, frame_image):
    # Data structures to store the RGB values of the pixels that will be analysed in the detection function. 
    hole_ahead = []
    ten_pixels_ahead = []
    # Convert the frame image to RGB in order to find what colours are appraoching based on an (x,y) coordinate system. 
    rgb = frame_image.convert('RGB')
    y_pos = mario_coords[0]

    #The first 10 pixels, from a 10 pixel offset from the position of the tip of Mario's cap, are collected. 
    pixel_positon_1 = mario_coords[1] + 10 
    pixel_positon_2 = mario_coords[1] + 11
    pixel_positon_3 = mario_coords[1] + 12
    pixel_positon_4 = mario_coords[1] + 13
    pixel_positon_5 = mario_coords[1] + 14
    pixel_positon_6 = mario_coords[1] + 15 
    pixel_positon_7 = mario_coords[1] + 16
    pixel_positon_8 = mario_coords[1] + 17
    pixel_positon_9 = mario_coords[1] + 18
    pixel_positon_10 = mario_coords[1] + 19

    # Five pixels will be gathered to determine if a hole is approaching, with the same 10 pixel offset to Mario's cap.
    hole_pixel_positon_1 = mario_coords[1] + 10 
    hole_pixel_positon_2 = mario_coords[1] + 11
    hole_pixel_positon_3 = mario_coords[1] + 12
    hole_pixel_positon_4 = mario_coords[1] + 13
    hole_pixel_positon_5 = mario_coords[1] + 14

    #This value will be the y-axis pixel location to determine if the floor is a hole (the RGB value will be the same as the sky colour value).
    hole = 210
    # Boolean value used to inform the detection function if a hole has been detected
    upcoming_hole = False

    # Gather the RGB pixels for the first 10 pixels infront of Mario with the 10 pixel offset to Mario's cap. 
    r1,g1,b1 = rgb.getpixel((pixel_positon_1,y_pos))
    r2,g2,b2 = rgb.getpixel((pixel_positon_2,y_pos))
    r3,g3,b3 = rgb.getpixel((pixel_positon_3,y_pos))
    r4,g4,b4 = rgb.getpixel((pixel_positon_4,y_pos))
    r5,g5,b5 = rgb.getpixel((pixel_positon_5,y_pos))
    r6,g6,b6 = rgb.getpixel((pixel_positon_6,y_pos))
    r7,g7,b7 = rgb.getpixel((pixel_positon_7,y_pos))
    r8,g8,b8 = rgb.getpixel((pixel_positon_8,y_pos))
    r9,g9,b9 = rgb.getpixel((pixel_positon_9,y_pos))
    r10,g10,b10 = rgb.getpixel((pixel_positon_10,y_pos))

    # Gather the RGB values for the first five pixels infront of Mario with the 10 pixel offset to Mario's cap and the hole y-axis value. 
    hr1,hg1,hb1 = rgb.getpixel((hole_pixel_positon_1,hole))
    hr2,hg2,hb2 = rgb.getpixel((hole_pixel_positon_2,hole))
    hr3,hg3,hb3 = rgb.getpixel((hole_pixel_positon_3,hole))
    hr4,hg4,hb4 = rgb.getpixel((hole_pixel_positon_4,hole))
    hr5,hg5,hb5 = rgb.getpixel((hole_pixel_positon_5,hole))

    # Check if all of the hole pixels are equal to the RGB values of the sky (R = 104, G = 136, B = 252) to determine if here is a hole approaching.
    if(hr1 == 104 and hg1 == 136 and hb1 == 252 or hr2 == 104 and hg2 == 136 and hb2 == 252 or hr3 == 104 and hg3 == 136 and hb3 == 252 or hr4 == 104 and hg4 == 136 and hb4 == 252 or hr5 == 104 and hg5 == 136 and hb5 == 252 ):
        # Appends all hole pixels, and returns this data structure
        upcoming_hole = True
        hole_ahead.append((hr1,hg1,hb1))
        hole_ahead.append((hr2,hg2,hb2))
        hole_ahead.append((hr3,hg3,hb3))
        hole_ahead.append((hr4,hg4,hb4))
        hole_ahead.append((hr5,hg5,hb5))
        hole_ahead.append(upcoming_hole)
        return hole_ahead

    #If there is no hole, append the first ten pixels and return the information
    ten_pixels_ahead.append((r1,g1,b1))
    ten_pixels_ahead.append((r2,g2,b2))
    ten_pixels_ahead.append((r3,g3,b3))
    ten_pixels_ahead.append((r4,g4,b4))
    ten_pixels_ahead.append((r5,g5,b5))
    ten_pixels_ahead.append((r6,g6,b6))
    ten_pixels_ahead.append((r7,g7,b7))
    ten_pixels_ahead.append((r8,g8,b8))
    ten_pixels_ahead.append((r9,g9,b9))
    ten_pixels_ahead.append((r10,g10,b10))
    ten_pixels_ahead.append(upcoming_hole)
    return ten_pixels_ahead

# Parameters: Mario's x and y coordinates based on the right-most red pixel on the tip of his cap
# Paramters:The current frame image. 
# This function determines the upcoming obstacles and enemies, and defines and executes actions depending on the environment
def detection(mario_coords, frame_image):
    #Defines the first 10 RGB values for the goomba enemy 
    goomba = [(104, 136, 252), (104, 136, 252), (104, 136, 252), (104, 136, 252), (104, 136, 252), (104, 136, 252), (228, 92, 16), (228, 92, 16), (228, 92, 16), (228, 92, 16)]

    #Defines the first 10 RGB values for the koopa enemy 
    koopa = [(104, 136, 252), (104, 136, 252), (104, 136, 252), (104, 136, 252), (252, 160, 68), (0, 168, 0), (252, 160, 68), (252, 160, 68), (252, 160, 68), (252, 160, 68)]

    #Defines the first 10 RGB values for the pipe object of any size
    pipe = [(184, 248, 24), (184, 248, 24), (184, 248, 24), (184, 248, 24), (0, 168, 0), (184, 248, 24), (184, 248, 24), (0, 168, 0), (0, 168, 0), (0, 168, 0)] 

    #Defines the first 10 RGB values for the solid block object
    solid_block = [(240, 208, 176), (240, 208, 176), (240, 208, 176), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (240, 208, 176), (240, 208, 176), (240, 208, 176)]

    # Defines the first 10 RGB values for the block just one step below the top of the solid block object
    solid_block_top = [(240, 208, 176), (240, 208, 176), (240, 208, 176), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (104, 136, 252), (104, 136, 252), (104, 136, 252)]

    #Gather environment information
    upcoming_pixels = information(mario_coords[0], frame_image)
    
    # Actions are allocated within these conditionals
    # If the last value in the upcoming_pixels list is false, this means that a hole is NOT approaching. 
    if(upcoming_pixels[-1] == False):
        # If the first and last pixels match from the upcoming pixel and goomba lists, then respond accordingly in the main function. 
        if(upcoming_pixels[0] == goomba[0] and upcoming_pixels[-2] == goomba[-1]): 
            return "Goomba"
        # If the upcoming pixel list matches the pipe list, then respond accordingly in the main function, this was done so that mario does not get confused with the bushes of the same colour. 
        if(upcoming_pixels[0:-1] == pipe):
            frame_image.save("test.png")
            return "Pipe"
        # If the first and last pixels match from the upcoming pixel and koopa lists, then respond accordingly in the main function. 
        if(upcoming_pixels[0] == koopa[0] and upcoming_pixels[-2] == koopa[-1]):
            return "Koopa"
        # If the first and last pixels match from the upcoming pixel and solid block lists, then respond accordingly in the main function. 
        if(upcoming_pixels[0] == solid_block[0] and upcoming_pixels[-2] == solid_block[-1]):
            return "Solid Block"
        # If the first and last pixels match from the upcoming pixel and solid block top lists, then respond accordingly in the main function. 
        if(upcoming_pixels[0] == solid_block_top[0] and upcoming_pixels[-2] == solid_block_top[-1]):
            return "Solid Block Top"
    else:
        return "Hole"
   
    
# Parameters: Observation array
#Function gathers Mario's (x,y) locations for the colour red on his clothes (248,56,0)
#Returns any upcoming obstacles
def upcoming_obstacle(obs):
    frame = Image.fromarray(obs)
    # Gets the pixel locations  in format (y,x) for Mario and returns mario's distance and the distance of the closest object. 
    mario_pixels = [i for i, pixel in enumerate(frame.getdata()) if pixel == (248,56,0)]
    width, height = frame.size
    #Gathers the pixel coordinates for each pixel that corresponds to the colour above. 
    mario_pixel_coords = [divmod(index, width) for index in mario_pixels]
    # This returns the furthest red pixel (right-most) that is attached to Mario's clothing. This represents the tip of his cap when facing right
    mario_pixel_coords.sort(key = lambda x: x[1] , reverse=True)
    result = detection(mario_pixel_coords, frame)
    return result
    

#Returns the performance matrix for this agent
def evaluation(info):
    performance_matrix = []
    pair = ()
    # The basic values have been extracted from the info dictionary and have been placed within the performance matrix. 
    for key, value in info.items():
        pair = (key,value)
        performance_matrix.append(pair)
    return performance_matrix

#Function used to evaluate Mario's performance on the stage
def Result_Return(performance_matrix, result, total_moves):
    final_x_pos=performance_matrix[8][1]
    flag_x_pos=3161
    Percentage_Complete = int((final_x_pos/flag_x_pos)*100)
    final_result = ""
    separator = "--------------------------------------------"
    #Assuming the flag is always at the same X-position, then 
    if(result == "Success"):
        final_result = "Performance Matrix: \n Coins: {}\n {}\n Flag_get: {}\n {}\n Life: {}\n {}\n Score: {}\n {}\n Stage: {}\n {}\n Status: {}\n {}\n Time: {}\n {}\n World: {}\n {}\n X-Position: {}\n {}\n Y-Position: {}\n {}\n Total Moves: {}\n{}\n\n\n Additional Data: \n {}\n Percentage of Level Completion:{}%\n".format(performance_matrix[0][1],separator, performance_matrix[1][1],separator,performance_matrix[2][1],separator,performance_matrix[3][1],separator,performance_matrix[4][1],separator,performance_matrix[5][1],separator,performance_matrix[6][1],separator,performance_matrix[7][1],separator,performance_matrix[8][1],separator,performance_matrix[9][1],separator,total_moves, separator, separator,Percentage_Complete)
        return final_result
    if(result == "Dead"):
        final_result = "Performance Matrix: \n Coins: {}\n {}\n Flag_get: {}\n {}\n Life: {}\n {}\n Score: {}\n {}\n Stage: {}\n {}\n Status: {}\n {}\n Time: {}\n {}\n World: {}\n {}\n X-Position: {}\n {}\n Y-Position: {}\n {}\n Total Moves: {}\n{}\n\n\n Additional Data: \n {}\n Percentage of Level Completion:{}%\n".format(performance_matrix[0][1],separator, performance_matrix[1][1],separator,performance_matrix[2][1],separator,performance_matrix[3][1],separator,performance_matrix[4][1],separator,performance_matrix[5][1],separator,performance_matrix[6][1],separator,performance_matrix[7][1],separator,performance_matrix[8][1],separator,performance_matrix[9][1],separator,total_moves, separator, separator,Percentage_Complete)
        return final_result
    if(result == "Dead"):
        return final_result
    if(result == "Stuck"):
        final_result = "Performance Matrix: \n Coins: {}\n {}\n Flag_get: {}\n {}\n Life: {}\n {}\n Score: {}\n {}\n Stage: {}\n {}\n Status: {}\n {}\n Time: {}\n {}\n World: {}\n {}\n X-Position: {}\n {}\n Y-Position: {}\n {}\n Total Moves: {}\n{}\n\n\n Additional Data: \n {}\n Percentage of Level Completion:{}%\n".format(performance_matrix[0][1],separator, performance_matrix[1][1],separator,performance_matrix[2][1],separator,performance_matrix[3][1],separator,performance_matrix[4][1],separator,performance_matrix[5][1],separator,performance_matrix[6][1],separator,performance_matrix[7][1],separator,performance_matrix[8][1],separator,performance_matrix[9][1],separator,total_moves, separator, separator,Percentage_Complete)
        return final_result
    if(result == "Dead"):
        return final_result
def main():
    # Environment setup:
    # The level played by the AI is the first level within the Super Mario Bros video game. 
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    oberservation = env.reset()
    performance_matrix = []
    result = ""
    totalmoves=0
    # This variable will determine if the AI is still playing the game. 
    playing = True
    done = False
    action = env.action_space.sample()
    while playing: 
        if(not done):
            obs, reward, terminated, truncated, info = env.step(1)
            done = terminated or truncated

        if(done and info['flag_get'] == False and info['time'] != 0):
            info['life'] = 0
            print('\n##############################')
            print("######  Mario has died  ######")
            print('##############################\n')
            performance_matrix = evaluation(info)
            result = Result_Return(performance_matrix, "Dead", totalmoves)
            playing = False
            break
        if(done and info['flag_get'] == False and info['time'] == 0):
            print('\n############################################################')
            print("######  Mario got stuck trying to complete the level  ######")
            print('############################################################\n')
            performance_matrix = evaluation(info)
            result = Result_Return(performance_matrix, "Stuck", totalmoves)
            playing = False
            break
        totalmoves+=1

        flag = info['flag_get']

        obstacle = upcoming_obstacle(obs)
        if(flag):
            print('\n############################################')
            print("######  Mario has captured the flag!  ######")
            print('############################################\n')
            performance_matrix = evaluation(info)
            result = Result_Return(performance_matrix, "Success", totalmoves)
            playing = False

        #IF a GOOMBA is approaching, Mario will jump to the right 
        if( obstacle  == "Goomba"):
            action = 2
            if(done == False):
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                totalmoves+=1
        
        #IF a KOOPA is approaching, Mario will jump to the right 
        elif(obstacle  == "Koopa"):
            action = 2
            for i in range(2):
                if(done == False):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    totalmoves+=1

        #IF Mario is approaching a PIPE, he will do an extended jump while moving right to make it over.   
        elif( obstacle == "Pipe"): 
            action = 5
            for i in range(8):
                if(done == False):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    totalmoves+=1
            action = 2
            for i in range(18):
                if(done == False):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    totalmoves+=1

        #IF Mario is approaching a HOLE, he will do an extended jump while moving right to make it over.   
        elif(obstacle  == "Hole"): 
            action = 2
            for i in range(16):
                if(done == False):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    totalmoves+=1

        #IF Mario is approaching or on SOLID BLOCK steps, Mario will jump to the right 
        elif( obstacle  == "Solid Block"):
            action = 2
            for i in range(6):
                if(done == False):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    totalmoves+=1

        #IF Mario is one step bleow  TOP of the SOLID BLOCK, he will do an extended jump while moving right to make it over.   
        elif( obstacle  == "Solid Block Top"):
            action = 2
            for i in range(30):
                if(done == False):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    totalmoves+=1
           
    return result
print(main())    