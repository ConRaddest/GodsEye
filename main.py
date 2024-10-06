import re
import time
import pickle
import cv2
from datetime import datetime
import numpy as np
import base64
import signal
import time
import face_recognition
import threading
import asyncio
import cv2
import numpy as np
from fastapi import Response
from nicegui import Client, app, core, run, ui
import requests
from multiprocessing import Manager, Queue
import cv2
from ultralytics import YOLO
import numpy as np
import face_recognition
from cvzone import FaceDetectionModule
from collections import Counter
import pickle
import asyncio
import concurrent.futures
import threading
face_detector = FaceDetectionModule.FaceDetector()
import os

# Classes for various tables
class User:
    def __init__(self, username, name, surname, password, preferred_lighting, preferred_temp, priority):
        self.username = username # Used to login to the app, uniquely identifies each person
        self.name = name  # User's first name
        self.surname = surname  # User's surname
        self.password = password  # User's password (would be hashed in real applications)
        self.preferred_lighting = preferred_lighting  # User's preferred lighting
        self.preferred_temp = preferred_temp  # User's preferred temperature
        self.priority = priority  # User's priority level (1 to 9)

class Persona:
    def __init__(self, id, username, face_encodings, date_captured):
        self.id = id
        self.username = username
        self.face_encodings = face_encodings
        self.date_captured = date_captured

class Camera:
    def __init__(self, name, ip_address):
        self.name = name
        self.ip_address = ip_address  

# global variables for app wide control
global_users = []
global_present_users = []
global_cameras = []
global_personas = []
global_all_known_encodings = []
global_all_usernames = []

global_application_data = {
    'preferred_lighting': 2,
    'preferred_temperature': 20,
    'frame_buffer_size': 5,
    'guest_recheck_threshold': 10,
    'zero_person_detection_interval_delay': 5,
    'body_detection_confidence_level': 0.4,
    'face_detection_confidence_level': 0.8,
}

global_application_is_running = False
global_webcam_debug_mode = False

# Computer Vision Variables
detect_bodies_confidence_level = 0
global_background_application_thread = None
global_add_persona_thread = None

global_logged_in_user = None

# activate application variables
global_current_lighting = 0 # 1 -> Dim, 2 -> Normal, 3 -> Bright
global_current_temperature = 0

# Paths for permanant storage
users_path = 'data/users.pkl'
cameras_path = 'data/cameras.pkl'
personas_path = 'data/personas.pkl'
application_path = 'data/application.pkl'
dir = 'data/'

# Data Paths and Directories
model = YOLO("yolo_weights/yolov8n.pt")

# Clear ALL pickle files
# ---------- USE WITH CAUTION -----------
delete_all = False

def preprocess_personas():
    global global_all_known_encodings, global_all_usernames
    global_all_known_encodings = []
    global_all_usernames = []

    # Loop through global_personas and combine encodings for each username
    for persona in global_personas:    
        global_all_usernames.extend([persona.username] * len(persona.face_encodings))
        global_all_known_encodings.extend([inner_array[0] for inner_array in persona.face_encodings])
        
def __init__():
    global global_users
    global global_personas
    global global_cameras
    global global_application_data
    global global_current_lighting
    global global_current_temperature

    if not os.path.exists(users_path):
        with open(users_path, 'wb') as f:
            pickle.dump([], f)

    if not os.path.exists(cameras_path):
        with open(cameras_path, 'wb') as f:
            pickle.dump([], f)

    if not os.path.exists(personas_path):
        with open(personas_path, 'wb') as f:
            pickle.dump([], f)

    if not os.path.exists(application_path):
        with open(application_path, 'wb') as f:
            pickle.dump(global_application_data, f)

    with open(users_path, 'rb') as f:
        global_users = pickle.load(f)

    with open(cameras_path, 'rb') as f:
        global_cameras = pickle.load(f)

    with open(personas_path, 'rb') as f:
        global_personas = pickle.load(f)

    with open(application_path, 'rb') as f:
        global_application_data = pickle.load(f)
        
    global_current_lighting = global_application_data['preferred_lighting']
    global_current_temperature = global_application_data['preferred_temperature']

    preprocess_personas()

__init__()

def camera_is_connected_verbose(ip_address):
    try:
        # Set the timeout to 100 milliseconds (0.1 seconds)
        response = requests.get(f'http://{ip_address}/test_verbose', timeout=1)
        
        if response.status_code == 200:
            ui.notify(f'Camera at {ip_address} is online!', color='green')
            return True
        else:
            ui.notify(f'Camera at {ip_address} has incorrect firmware.', color='red')
            return False
    
    except requests.exceptions.Timeout:
        # Handle timeout error (when the camera is offline or unreachable)
        ui.notify(f'Camera at {ip_address} is offline.', color='red')
        return False
    
    except requests.exceptions.RequestException as e:
        # Handle other types of request errors
        ui.notify(f'Failed to connect to the camera: {e}', color='red')
        return False
    
def camera_is_connected(ip_address):
    try:
        response = requests.get(f'http://{ip_address}/test', timeout=1)
        if response.status_code == 200: return True
        else: return False
    except: return False

# Returns the visible usernames based on the incoming frame
def recognise_faces(frame):
    usernames = []
    input_face_locations = []
    # Extract face encodings from the incoming frame
    faces = face_detector.findFaces(frame, draw=False)[1]
    for face in faces:
        if face['score'][0] > global_application_data['face_detection_confidence_level']:
            x, y, w, h = face['bbox']

            face_location = [(y, x + w, y + h, x)]
            input_face_locations.append(face_location)
            
    if len(input_face_locations) < 1:
        return []
    
    input_face_encodings = face_recognition.face_encodings(frame, input_face_locations[0])

    # Loop through the encodings in the frame
    for unknown_encoding in input_face_encodings:
        # Compare the unknown face to all known face encodings in one go
        boolean_matches = face_recognition.compare_faces(global_all_known_encodings, unknown_encoding)
            
        # Use Counter to tally the matches for each username
        votes = Counter(face_id for match, face_id in zip(boolean_matches, global_all_usernames) if match)

        if votes:
            # Get the most common username from the matches
            best_match_username = votes.most_common(1)[0][0]
        else:
            best_match_username = "Unknown"

        # Append the identified username to the list
        usernames.append(best_match_username)
    return usernames

# Returns the visible number of bodies
def detect_number_bodies(frame):
    number_bodies = 0
    results = model(frame, stream=True, classes=0, conf=global_application_data['body_detection_confidence_level'], verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            number_bodies +=1

    return number_bodies

def update_led_status(led_status, ip_address):
    # -1 -> Turn Off Flash
    # 0 -> Turn On Flash

    # 1 -> Turn Lighting to Dim
    # 2 -> Turn Lighting to Normal
    # 3 -> Turn Lighting to Bright/Bright
    
    url = f"http://{ip_address}/control?led={led_status}"
    try:
        response = requests.get(url, timeout=1)
        if response.status_code != 200:
            print(f"Error: Received response code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

def update_led_status_all(led_status):
    # -1 -> Turn Off Flash
    # 0 -> Turn On Flash

    # 1 -> Turn Lighting to Dim
    # 2 -> Turn Lighting to Normal
    # 3 -> Turn Lighting to Bright/Bright

    ip_addresses = []
    for camera in global_cameras:
        ip_addresses.append(camera.ip_address)
    
    urls = [f"http://{ip_address}/control?led={led_status}" for ip_address in ip_addresses]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code != 200:
                print(f"Error: Received response code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

def handle_persona_add(face_encodings):
    global global_personas, global_logged_in_user, global_users

    if not global_personas:
        new_persona_id = 1  # If no personas exist, start with ID 1
    else:
        # Get the highest existing ID and add 1
        new_persona_id = max(persona.id for persona in global_personas) + 1

    new_persona = Persona(
        id=new_persona_id,
        username=global_logged_in_user.username,
        face_encodings=face_encodings,
        date_captured=time.time(),
    )

    global_personas.append(new_persona)

    # Updating the store with the saved users
    with open(personas_path, 'wb') as f:
        pickle.dump(global_personas, f)

    preprocess_personas()
    return
        
def handle_user_add(user_data):
    global global_users

    new_user = User(
        username=user_data['username'],
        name=user_data['name'],
        surname=user_data['surname'],
        password=user_data['password'],
        preferred_lighting=user_data['preferred_lighting'],
        preferred_temp=user_data['preferred_temp'],
        priority=user_data['priority'],
    )

    global_users.append(new_user)

    # Updating the store with the saved users
    with open(users_path, 'wb') as f:
        pickle.dump(global_users, f)

    return

def handle_user_update(user_data):
    global global_users
    global global_logged_in_user

    updated_user = User(
        username=user_data['username'],
        name=user_data['name'],
        surname=user_data['surname'],
        password=user_data['password'],
        preferred_lighting=user_data['preferred_lighting'],
        preferred_temp=user_data['preferred_temp'],
        priority=user_data['priority'],
    )

    index = global_users.index(global_logged_in_user)

    global_users[index] = updated_user

    global_logged_in_user = updated_user

    # Updating the store with the saved users
    with open(users_path, 'wb') as f:
        pickle.dump(global_users, f)

    return

def handle_application_settings_update(application_data):
    global global_application_data

    global_application_data = {
        'preferred_lighting': application_data['default_lighting_input'],
        'preferred_temperature': application_data['default_temperature_input'],
        'frame_buffer_size': application_data['frame_buffer_size_input'],
        'guest_recheck_threshold': application_data['guest_recheck_threshold_input'],
        'zero_person_detection_interval_delay': application_data['zero_person_detection_interval_delay_input'],
        'body_detection_confidence_level': application_data['body_detection_confidence_level_input'],
        'face_detection_confidence_level': application_data['face_detection_confidence_level_input']
    }

    # Updating the store with the saved users
    with open(application_path, 'wb') as f:
        pickle.dump(global_application_data, f)

    return

def handle_login(username, password):
    global global_users
    global global_logged_in_user

    for user in global_users:
        if user.username == username and user.password == password:
            ui.navigate.to('/dashboard')  # Redirect to the dashboard page on successful login
            global_logged_in_user = user
            return
    ui.notify('Invalid username or password!', color='red')

# Login page layout
def login_page():
    if global_logged_in_user:
        ui.navigate.to('/dashboard')
        return
    
    with ui.card().classes('w-full sm:w-3/4 md:w-1/2 lg:w-1/4 mx-auto mt-20 p-10'):
        ui.label('Login').classes('text-3xl mb-5 text-center')
        username_input = ui.input(label='Username').classes('mb-5 w-full')
        password_input = ui.input(label='Password', password=True).classes('mb-5 w-full')

        def on_login_click():
            handle_login(username_input.value, password_input.value)

        ui.button('Login', on_click=on_login_click).classes('w-full')
        ui.button('Register', on_click=lambda: on_register_click(), color="blue").classes('w-full mt-2')

    def on_register_click():
        if len(global_users) == 5:
            ui.notify('Maximum number of users have been registered. Please consult the portal admin.', color='red')
            return
        
        ui.navigate.to('/register')

def logout():
    global global_logged_in_user
    global global_application_is_running
    global global_background_application_thread

    global_logged_in_user = None
    global_application_is_running = False

    if global_background_application_thread:
        global_background_application_thread.join()

    ui.navigate.to('/')
    
# Registration page layout
def registration_page():
    if global_logged_in_user:
        ui.navigate.to('/dashboard')
        return
    with ui.card().classes('w-full sm:w-3/4 md:w-1/2 lg:w-1/2 mx-auto'):
        ui.label('Register').classes('text-3xl mb-5 text-center')
        
        username_input = ui.input(label='Username*').classes('mb-5 w-full')

        if len(global_users) == 0:
            username_input.disable()
            username_input.value = 'admin'

        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            name_input = ui.input(label='Name*', validation={'Invalid characters used': lambda value: bool(re.match(r'^[A-Za-z0-9 ]*$', value))}).classes('w-full')
            surname_input = ui.input(label='Surname*', validation={'Invalid characters used': lambda value: bool(re.match(r'^[A-Za-z0-9 ]*$', value))}).classes('w-full')

        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            password_input = ui.input(label='New Password*', password=True, password_toggle_button=True).classes('w-full')
            confirm_password_input = ui.input(label='Confirm New Password*', password=True, password_toggle_button=True, validation={'Passwords do not match': lambda value: value == password_input.value}).classes('w-full')

        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            preferred_lighting_input = ui.select({1: 'Dim', 2: 'Normal', 3: 'Bright'}, value=2, label='Select Preferred Lighting').classes('w-full')
            
            taken_priorities = [user.priority for user in global_users]
            available_priorities = [priority for priority in range(1, 10) if priority not in taken_priorities]

            priority_input = ui.select(available_priorities, label='Select Priority (Higher = Important)', value=available_priorities[len(available_priorities)-1]).classes('w-full')

        ui.label('Select Preffered Temp (C): ').props('style="color: gray;"')
        preferred_temp_input = ui.slider(min=10, max=35, value=22).classes('mb-5 w-full').props('label-always')
        
        def on_register_click():
            user_data = {
                'username': username_input.value.strip(),
                'name': name_input.value.strip(),
                'surname': surname_input.value.strip(),
                'password': password_input.value.strip(),
                'preferred_lighting': preferred_lighting_input.value,
                'preferred_temp': preferred_temp_input.value,
                'priority': priority_input.value,
            }

            for user in global_users:
                if user_data['username'] == user.username:
                    ui.notify(f'Username Taken', color='red')
                    return
                if user_data['username'] == 'Unknown':
                    ui.notify(f'OS Reserved Username', color='red')
                    return

            if len(user_data['username']) == 0 or len(user_data['name']) == 0 or len(user_data['surname']) == 0 or len(user_data['password']) == 0:
                ui.notify(f'Missing Required Fields', color='red')
                return
            
            if user_data['password'] != confirm_password_input.value.strip():
                ui.notify(f'Passwords Do Not Match', color='red')
                return
            
            handle_user_add(user_data)
            ui.navigate.to('/')
                
            
        with ui.row().classes('w-full'):
            ui.button('Cancel', on_click=lambda: ui.navigate.to('/'), color="gray").classes('w-full m-0 text-white').style('flex: 1; margin-left: 0.01rem;')
            ui.button('Register', on_click=on_register_click, color="blue").classes('w-full m-0').style('flex: 1; margin-right: 0.01rem;')

# ------------------------------- PROTECTED PAGES (USER MUST BE LOGGED IN) ---------------------------------------
def dashboard_page():
    def start_async_task():
        asyncio.run(main_application())

    def start_application():
        if len(global_personas) == 0: 
            ui.notify('No Personas Added', color='red')
            return
        
        # Checking if we have cameras
        if len(global_cameras) == 0 and not global_webcam_debug_mode: 
            ui.notify('No Cameras Added', color='red')
            return
        
        # Checking if all cameras are correctly setup and connected
        for camera in global_cameras:
            if not camera_is_connected_verbose(camera.ip_address):
                return

        global global_application_is_running
        global global_background_application_thread

        if not global_application_is_running:
            global_application_is_running = True
            
            # Create a thread to run the asyncio loop
            global_background_application_thread = threading.Thread(target=start_async_task, daemon=True)
            global_background_application_thread.start()
            
        toggle_button_ui.refresh()
        ui.notify('Application Running', color='green')

    def stop_application():
        global global_application_is_running, global_current_lighting, global_current_temperature, global_present_users
        if global_application_is_running:
            global_application_is_running = False

        # Resetting back to defaults if application is not running, removing all present users
        update_dashboard([])
        
        global_current_lighting = global_application_data['preferred_lighting']
        global_current_temperature = global_application_data['preferred_temperature']

        dashboard_cards.refresh()
        toggle_button_ui.refresh()

    def update_dashboard(present_usernames):
        global global_present_users, global_current_lighting, global_current_temperature, global_application_data

        global_present_users = [user for user in global_users if user.username in present_usernames]
        
        if len(global_present_users) > 0:
            global_present_users.sort(key=lambda user: user.priority, reverse=True)
            global_current_lighting = global_present_users[0].preferred_lighting
            global_current_temperature = global_present_users[0].preferred_temp

            update_led_status_all(global_current_lighting)

        else: 
            # there's no one we know so setting to default values
            global_current_lighting = global_application_data['preferred_lighting']
            global_current_temperature = global_application_data['preferred_temperature']

            update_led_status_all(global_current_lighting)

        dashboard_cards.refresh()

    async def main_application():
        global global_application_data
        ip_addresses = []
        
        for camera in global_cameras:
            ip_addresses.append(camera.ip_address)

        # -------------- Single Camera Algorithm ---------------
        if len(ip_addresses) == 1 or global_webcam_debug_mode:
            # Counters
            frame_counter = 0
            guest_recheck_counter = 0
            exit_delay_counter = 0

            # Calculated Variables
            number_bodies = []
            present_usernames = []

            average_number_bodies = 0
            prev_average_number_bodies = 0

            # Flags
            avg_num_bodies_changed = False
            unknown_resident_present = False
            
            if global_webcam_debug_mode:
                url = 0
            else:
                url = 'http://' + ip_addresses[0] + ':81'
            
            video = cv2.VideoCapture(url)

            while global_application_is_running:
                ret, frame = video.read()
                if ret:
                    frame_counter += 1
                    
                    # Detecting how many people and faces in the room
                    number_bodies.append(detect_number_bodies(frame))
                    
                    if frame_counter % global_application_data['frame_buffer_size'] == 0:
                        # Calculating average # people in the room
                        average_number_bodies = round(sum(number_bodies)/len(number_bodies))
                        # Resetting Buffer
                        number_bodies = []
                        
                        # Number of people has changed and someone is inside the room    
                        if (prev_average_number_bodies != average_number_bodies) and (average_number_bodies != 0):
                            avg_num_bodies_changed = True
                            present_usernames = []

                        # No one is inside the room, resetting alles after exit delay
                        if average_number_bodies == 0:
                            exit_delay_counter += 1

                        # No one is inside the room, resetting alles
                        if exit_delay_counter > global_application_data['zero_person_detection_interval_delay']:
                            avg_num_bodies_changed = False
                            present_usernames = []
                            guest_recheck_counter = 0
                            unknown_resident_present = False
                            update_dashboard(present_usernames)
                            exit_delay_counter = 0

                        # Updating Tracker
                        prev_average_number_bodies = average_number_bodies 

                        # ------------- Logging for Debugging ---------------
                        print('Avg # People: ', average_number_bodies)
                        print('Present Usernames: ', present_usernames)

                        if avg_num_bodies_changed: print('Avg # People Changed!')
                        if unknown_resident_present: print("Unknown Resident Present!") 

                    # ----------------- If number of people have changed, or if there is an unknown present ------------------
                    if (avg_num_bodies_changed and frame_counter % global_application_data['frame_buffer_size'] == 0 
                        or (unknown_resident_present
                        and frame_counter % global_application_data['frame_buffer_size'] == 0 
                        and guest_recheck_counter < global_application_data['guest_recheck_threshold'])):

                        usernames_in_frame = recognise_faces(frame)

                        unique_present_usernames = list(set(present_usernames + usernames_in_frame))
                        known_usernames_in_present_list = [username for username in unique_present_usernames if username != "Unknown"]

                        if len(known_usernames_in_present_list) == average_number_bodies: 
                            # --- Everyone has been identified ---
                            unique_present_usernames = known_usernames_in_present_list
                            unknown_resident_present = False
                            avg_num_bodies_changed = False
                            exit_delay_counter = 0
            
                        elif len(known_usernames_in_present_list) < average_number_bodies:
                            # --- There is still people we do not know ---
                            unique_present_usernames = known_usernames_in_present_list
                            while len(unique_present_usernames) < average_number_bodies:
                                unique_present_usernames.append('Unknown')

                            avg_num_bodies_changed = False
                            unknown_resident_present = True
                            guest_recheck_counter += 1
                            exit_delay_counter = 0

                            print('# Guest Rechecks: ', guest_recheck_counter)

                        present_usernames = unique_present_usernames
                        update_dashboard(unique_present_usernames)

        # -------------- Dual Camera Algorithm ---------------
        if len(ip_addresses) == 2:
            # Counter
            frame_counter = 0
            guest_recheck_counter = 0
            exit_delay_counter = 0

            # Calculated Variables
            number_bodies1 = []
            number_bodies2 = []
            present_usernames = []

            average_number_bodies = 0
            prev_average_number_bodies = 0

            # Flags
            avg_num_bodies_changed = False
            unknown_resident_present = False

            video1 = cv2.VideoCapture('http://' + ip_addresses[0] + ':81')
            video2 = cv2.VideoCapture('http://' + ip_addresses[1] + ':81')

            while global_application_is_running:
                # Assuming two cameras per room, need to implement second camera
                ret1, frame1 = video1.read()
                ret2, frame2 = video2.read()

                if ret1 and ret2:
                    frame_counter += 1
                    # Detecting how many people are in the room
                    number_bodies1.append(detect_number_bodies(frame1))
                    number_bodies2.append(detect_number_bodies(frame1))

                    # number_faces = detect_number_faces(frame)
                    
                    if frame_counter % global_application_data['frame_buffer_size'] == 0:
                        # Calculating average # people in the room, taking maximum from each average
                        average_number_bodies1 = round(sum(number_bodies1)/len(number_bodies1))
                        average_number_bodies2 = round(sum(number_bodies1)/len(number_bodies1))           
                        average_number_bodies = max([average_number_bodies1, average_number_bodies2])

                        # Resetting Buffer
                        number_bodies1 = []
                        number_bodies2 = []

                        # Number of people has changed and someone is inside the room    
                        if (prev_average_number_bodies != average_number_bodies) and (average_number_bodies != 0):
                            avg_num_bodies_changed = True
                            present_usernames = []

                        # No one is inside the room, resetting alles after exit delay
                        if average_number_bodies == 0:
                            exit_delay_counter += 1

                        if exit_delay_counter > global_application_data['zero_person_detection_interval_delay']:
                            avg_num_bodies_changed = False
                            present_usernames = []
                            guest_recheck_counter = 0
                            unknown_resident_present = False
                            update_dashboard(present_usernames)
                            exit_delay_counter = 0

                        # Updating Tracker
                        prev_average_number_bodies = average_number_bodies

                        # ------------- Logging for Debugging ---------------
                        print('Avg # People: ', average_number_bodies)
                        print('Present Usernames: ', present_usernames)
                        

                        if avg_num_bodies_changed: print('Avg # People Changed!')
                        if unknown_resident_present: print("Unknown Resident Present!")        

                    # ----------------- If number of people have changed, or if there is an unknown present ------------------
                    if (avg_num_bodies_changed and frame_counter % global_application_data['frame_buffer_size'] == 0 
                        or (unknown_resident_present
                        and frame_counter % global_application_data['frame_buffer_size'] == 0 
                        and guest_recheck_counter < global_application_data['guest_recheck_threshold'])):

                        # Getting who can be seen in each frame
                        usernames_in_frame1 = recognise_faces(frame1)
                        usernames_in_frame2 = recognise_faces(frame2)

                        # Extracting unique usernames from those two lists
                        unique_present_usernames = list(set(present_usernames + usernames_in_frame1 + usernames_in_frame2))
                        # Removing the unidentified usernames from the list
                        known_usernames_in_present_list = [username for username in unique_present_usernames if username != "Unknown"]

                        if len(known_usernames_in_present_list) == average_number_bodies: 
                            # --- Everyone has been identified ---
                            unique_present_usernames = known_usernames_in_present_list
                            unknown_resident_present = False
                            avg_num_bodies_changed = False
                            exit_delay_counter = 0
            
                        elif len(known_usernames_in_present_list) < average_number_bodies:
                            # --- There is still people we do not know ---
                            unique_present_usernames = known_usernames_in_present_list
                            while len(unique_present_usernames) < average_number_bodies:
                                unique_present_usernames.append('Unknown')

                            avg_num_bodies_changed = False
                            unknown_resident_present = True
                            exit_delay_counter = 0

                            guest_recheck_counter += 1

                            print('# Guest Rechecks: ', guest_recheck_counter)

                        present_usernames = unique_present_usernames
                        update_dashboard(unique_present_usernames)
                
    with ui.left_drawer(bottom_corner=True).style('background-color: #d7e3f4;') as menu_drawer:
        ui.button('Dashboard', on_click=lambda: ui.navigate.to('/dashboard'), icon='dashboard').style('font-size: 15px;').props('flat color=black')
        ui.button('Personas', on_click=lambda: ui.navigate.to('/personas'), icon='face').style('font-size: 15px;').props('flat color=black')
        ui.button('Cameras', on_click=lambda: ui.navigate.to('/cameras'), icon='camera').style('font-size: 15px;').props('flat color=black')
        ui.button('Residents', on_click=lambda: ui.navigate.to('/residents'), icon='list').style('font-size: 15px;').props('flat color=black')

    with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
        ui.button('Menu', on_click=lambda: menu_drawer.toggle(), icon='menu').props('flat color=white')
        ui.button('Logout', on_click=lambda: logout(), icon='logout').props('flat color=white')
        ui.button('Actions', on_click=lambda: actions_drawer.toggle(), icon='splitscreen').props('flat color=white')

    with ui.right_drawer(bottom_corner=True).style('background-color: #d7e3f4;') as actions_drawer:
        @ui.refreshable
        def toggle_button_ui():
            toggle_button_text = ''
            toggle_button_icon = ''

            if global_application_is_running: toggle_button_text = 'Stop Application'; toggle_button_icon = 'stop'
            else: toggle_button_text = 'Start Application'; toggle_button_icon = 'play_arrow'
        
            ui.button(toggle_button_text, on_click=lambda: toggle_application(), icon=toggle_button_icon).props('flat color=black')

        toggle_button_ui()
        ui.button('Profile', on_click=lambda: update_profile_dialog.open(), icon='person').props('flat color=black')

        ui.space()
        if global_logged_in_user.username == 'admin':
            ui.button('Settings', on_click=lambda: update_application_settings_dialog.open(), icon='settings').props('flat color=black')

        def toggle_application():
            if global_application_is_running:
                stop_application()
                ui.notify('Application Stopped', color='green')
            else:
                start_application()

        actions_drawer.toggle()

    @ui.refreshable
    def dashboard_cards():
        with ui.card().classes('w-full') as dashboard_card:
            ui.label('Dashboard').classes('text-3xl mb-1')
            with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
                with ui.card().classes('mb-2 p-2 h-[150px] break-inside-avoid flex justify-center items-center'):
                    
                    if global_current_lighting == 1: 
                        current_lighting = 'Dim'
                        current_lighting_icon = 'trip_origin'
                        current_icon_size = 'text-5xl'
                        current_icon_class = 'py-3'
                    elif global_current_lighting == 2: 
                        current_lighting = 'Normal'
                        current_lighting_icon = 'o_light_mode'
                        current_icon_size = 'text-7xl'
                        current_icon_class = ''
                    else: 
                        current_lighting = 'Bright'
                        current_lighting_icon = 'light_mode'
                        current_icon_size = 'text-7xl'
                        current_icon_class = ''
     
                    ui.icon(current_lighting_icon).classes(current_icon_size).classes(current_icon_class)  # Larger icon size

                    ui.label(current_lighting + ' Lighting').classes('text-lg mt-2 text-center')

                with ui.card().classes('mb-2 p-2 h-[150px] break-inside-avoid flex justify-center items-center'):
                    ui.label(str(global_current_temperature) + '°C').classes('text-6xl text-center') 
                    ui.label('Current Temperature').classes('text-lg mt-2 text-center')

        with ui.row().classes('w-full'):
            with ui.card().classes('w-full'):
                ui.label('Residents Currently Present').classes('w-full text-xl')
                columns = [
                    {'name': 'name', 'label': 'Name', 'field': 'name', 'required': True, 'align': 'left'},
                    {'name': 'preferred_lighting', 'label': 'Preferred Lighting', ':field': 'row => row.preferred_lighting == 1 ? "Dim" : row.preferred_lighting == 2 ? "Normal" : row.preferred_lighting == 3 ? "Bright" : "Unknown"', 'align': 'left'},
                    {'name': 'preferred_temp', 'label': 'Preferred Temp', 'field': 'preferred_temp', 'align': 'left'},
                    {'name': 'priority', 'label': 'Priority Level', 'field': 'priority', 'sortable': True, 'align': 'left'},
                ]

                # Create rows from global_users list
                rows = [
                    {
                        'name': user.name + ' ' + user.surname,
                        'preferred_lighting': user.preferred_lighting,
                        'preferred_temp': str(user.preferred_temp) + "°C",
                        'priority': user.priority,
                    }
                    for user in global_present_users
                ]
                
                # Create a table to display the residents
                table = ui.table(columns=columns, rows=rows, row_key='username').classes('w-full mx-auto p-3')

                table.add_slot('body-cell-priority', '''
                    <q-td key="priority" :props="props">
                        <q-badge :color="
                            props.value == 1 ? 'green' :
                            props.value <= 3 ? 'light-green' :
                            props.value <= 5 ? 'yellow' :
                            props.value <= 7 ? 'orange' : 'red'">
                            {{ props.value }}
                        </q-badge>
                    </q-td>
                ''')
    dashboard_cards()

    with ui.dialog() as update_profile_dialog, ui.card().classes('w-full sm:w-3/4 md:w-1/2 lg:w-1/2 mx-auto'):
        ui.label('Update Profile').classes('text-3xl mb-5 text-center')
        
        username_input = ui.input(label='Username*').classes('mb-5 w-full')

        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            name_input = ui.input(label='Name*', validation={'Invalid characters used': lambda value: bool(re.match(r'^[A-Za-z0-9 ]*$', value))}).classes('w-full')
            surname_input = ui.input(label='Surname*', validation={'Invalid characters used': lambda value: bool(re.match(r'^[A-Za-z0-9 ]*$', value))}).classes('w-full')

        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            password_input = ui.input(label='New Password*', password=True, password_toggle_button=True).classes('mb-5 w-full')
            confirm_password_input = ui.input(label='Confirm New Password*', password=True, password_toggle_button=True, validation={'Passwords do not match': lambda value: value == password_input.value}).classes('mb-5 w-full')

        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            preferred_lighting_input = ui.select({1: 'Dim', 2: 'Normal', 3: 'Bright'}, label='Select Preferred Lighting').classes('w-full')
            
            taken_priorities = [user.priority for user in global_users if user.priority != global_logged_in_user.priority]
            available_priorities = [priority for priority in range(1, 10) if priority not in taken_priorities]
            
            priority_input = ui.select(available_priorities, label='Select Priority (Higher = Important)').classes('mb-5 w-full mx-auto').props('inline')

        ui.label('Select Preffered Temp (C): ').props('style="color: gray;"')
        preferred_temp_input = ui.slider(min=10, max=35).classes('mb-5 w-full').props('label-always')

        username_input.value = global_logged_in_user.username

        name_input.value = global_logged_in_user.name
        surname_input.value = global_logged_in_user.surname

        password_input.value = global_logged_in_user.password
        confirm_password_input.value = global_logged_in_user.password

        preferred_lighting_input.value = global_logged_in_user.preferred_lighting
        preferred_temp_input.value = global_logged_in_user.preferred_temp
        priority_input.value = global_logged_in_user.priority
        
        def on_update_click():
            user_data = {
                'username': username_input.value.strip(),
                'name': name_input.value.strip(),
                'surname': surname_input.value.strip(),
                'password': password_input.value.strip(),
                'preferred_lighting': preferred_lighting_input.value,
                'preferred_temp': preferred_temp_input.value,
                'priority': priority_input.value,
            }

            for user in global_users:
                if global_logged_in_user.username != user.username and user_data['username'] == user.username:
                    ui.notify(f'Username Taken', color='red')
                    return
                if user_data['username'] == 'Unknown':
                    ui.notify(f'OS Reserved Username', color='red')
                    return

            if len(user_data['username']) == 0 or len(user_data['name']) == 0 or len(user_data['surname']) == 0 or len(user_data['password']) == 0:
                ui.notify(f'Missing Required Fields', color='red')
                return
            
            if user_data['password'] != confirm_password_input.value.strip():
                ui.notify(f'Passwords Do Not Match', color='red')
                return
            
            handle_user_update(user_data)
            update_profile_dialog.close()
            
            update_dashboard([user.username for user in global_present_users])
            dashboard_cards.refresh()
                
        with ui.row().classes('w-full'):
            ui.button('Cancel', on_click=lambda: update_profile_dialog.close(), color="gray").classes('w-full m-0 text-white').style('flex: 1; margin-left: 0.01rem;')
            ui.button('Update', on_click=on_update_click, color="blue").classes('w-full m-0').style('flex: 1; margin-right: 0.01rem;')
    
    with ui.dialog() as update_application_settings_dialog, ui.card().classes('w-full sm:w-3/4 md:w-1/2 lg:w-1/2 mx-auto'):        
        ui.label('Edit Application Settings').classes('text-3xl')
        ui.label('Some of these settings can drastically affect the performance of the application. Only alter if you understand the implications.').classes('text-md justify-start')
        
        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            
            with ui.label('Select Default Lighting').props('style="color: gray;"'):
                ui.tooltip('Default lighting for empty rooms, and rooms with unknown people.')
            default_lighting_input = ui.select({1: 'Dim', 2: 'Normal', 3: 'Bright'}, value=global_application_data['preferred_lighting']).classes('w-full mb-5')
            
        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            with ui.label('Select Default Temp (C)').props('style="color: gray;"').classes('mb-10'):
                ui.tooltip('Default temperature for empty rooms, and rooms with unknown people.')
            default_temperature_input = ui.slider(min=10, max=35, value=global_application_data['preferred_temperature']).classes('mb-5').props('label-always')
    
        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            with ui.label('Select Frame Buffer Size').props('style="color: gray;"').classes('mb-10'):
                ui.tooltip('Number of frames that calculate average number of people within a room balacing speed vs. accuracy. \n\n Also determines the interval for face recognition. ')
            frame_buffer_size_input = ui.slider(min=1, max=20, value=global_application_data['frame_buffer_size']).classes('mb-5').props('label-always')

        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            with ui.label('Select Guest Recheck Threshold').props('style="color: gray;"').classes('mb-10'):
                ui.tooltip('Number of times Unknown faces should be reidentified to negate false negatives.')
            guest_recheck_threshold_input = ui.slider(min=1, max=20, value=global_application_data['guest_recheck_threshold']).classes('mb-5').props('label-always')
        
        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):
            with ui.label('Zero Person Detection Delay Interval').props('style="color: gray;"').classes('mb-10'):
                ui.tooltip('Number of frame buffer intervals before the room is deemed empty.')
            zero_person_detection_interval_delay_input = ui.slider(min=1, max=10, value=global_application_data['zero_person_detection_interval_delay']).classes('mb-5').props('label-always')
            
        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):  
            with ui.label('Select Body Detection Confidence Threshold').props('style="color: gray;"').classes('mb-10'):
                ui.tooltip('Confidence level for body detection, higher = stricter. Used in persence detection algorithm.')
            body_detection_confidence_level_input = ui.slider(min=0.1, max=1, step=0.05, value=global_application_data['body_detection_confidence_level']).classes('mb-5').props('label-always')
        
        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):  
            with ui.label('Select Face Detection Confidence Threshold').props('style="color: gray;"').classes('mb-10'):
                ui.tooltip('Confidence level for face detection, higher = stricter. User for persona registration and face recognition.')
            face_detection_confidence_level_input = ui.slider(min=0.1, max=1, step=0.05, value=global_application_data['face_detection_confidence_level']).classes('mb-5').props('label-always')

        with ui.element('div').classes('sm:columns-1 md:columns-2 lg:columns-2 w-full gap-2'):  
            with ui.label('Enable Webcam Debug Mode').props('style="color: gray;"').classes('mb-10'):
                ui.tooltip('Enables use of webcam instead of registered cameras, useful for debugging and testing.')
            webcam_debug_mode_input = ui.checkbox(value=global_webcam_debug_mode).classes('mb-5')

        
        def on_update_click():
            application_data = {
                'default_lighting_input': default_lighting_input.value,
                'default_temperature_input': default_temperature_input.value,
                'frame_buffer_size_input': frame_buffer_size_input.value,
                'guest_recheck_threshold_input': guest_recheck_threshold_input.value,
                'zero_person_detection_interval_delay_input':zero_person_detection_interval_delay_input.value,
                'body_detection_confidence_level_input': body_detection_confidence_level_input.value,
                'face_detection_confidence_level_input': face_detection_confidence_level_input.value,
            }
            
            global global_webcam_debug_mode

            global_webcam_debug_mode = webcam_debug_mode_input.value

            handle_application_settings_update(application_data)

            update_dashboard([user.username for user in global_present_users])
            update_application_settings_dialog.close()
                
            
        with ui.row().classes('w-full'):
            ui.button('Cancel', on_click=lambda: update_application_settings_dialog.close(), color="gray").classes('w-full m-0 text-white').style('flex: 1; margin-left: 0.01rem;')
            ui.button('Update', on_click=on_update_click, color="blue").classes('w-full m-0').style('flex: 1; margin-right: 0.01rem;')

def personas_page():
    with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
        ui.button('Menu', on_click=lambda: menu_drawer.toggle(), icon='menu').props('flat color=white')
        ui.button('Logout', on_click=lambda: logout(), icon='logout').props('flat color=white')
    
    with ui.left_drawer(bottom_corner=True).style('background-color: #d7e3f4;') as menu_drawer:
        ui.button('Dashboard', on_click=lambda: ui.navigate.to('/dashboard'), icon='dashboard').style('font-size: 15px;').props('flat color=black')
        ui.button('Personas', on_click=lambda: ui.navigate.to('/personas'), icon='face').style('font-size: 15px;').props('flat color=black')
        ui.button('Cameras', on_click=lambda: ui.navigate.to('/cameras'), icon='camera').style('font-size: 15px;').props('flat color=black')
        ui.button('Residents', on_click=lambda: ui.navigate.to('/residents'), icon='list').style('font-size: 15px;').props('flat color=black')

    with ui.card().classes('w-full mx-auto p-10') as personas_card:
        with ui.row().classes('justify-between items-center w-full'):
            ui.label('Personas').classes('text-3xl mb-5 text-left')
            with ui.row().classes('justify-between items-center w-0.2'):
                with ui.button(on_click=lambda: on_click_persona_add('camera'), icon='camera', color='blue').style('font-size: 15px;'):
                    ui.tooltip('Use Surveilance Cameras').style('font-size: 15px;')
                with ui.button(on_click=lambda: on_click_persona_add('videocam'), icon='videocam', color='blue').style('font-size: 15px;'):
                    ui.tooltip('Use Webcam (Laptops Only)').style('font-size: 15px;')

        with ui.dialog().props('persistent') as add_persona_webcam_dialog, ui.card().classes('w-full sm:w-3/4 md:w-1/2 lg:w-1/2 mx-auto p-10'):
            ui.label('The application will now gather data of your face to recognise you. '
                    + 'During data collection, ensure to move your face into different poses, '
                    + 'angles and distances from the camera for best results, ensuring that there is no-one else in the frame. '
                    + 'During the process the application may hang or become unresponsive. '
                    + 'Do not exit or press any keys until the data collection process is complete.').classes('pb-2 text-justify').style('font-size: 15px;')
        
            ui.label('Select Number of Pictures: ').classes('mb-2 font-semibold text-center').style('font-size: 15px;') 
            number_pictures_webcam_input = ui.slider(min=1, max=40, value=20).classes('mb-5 w-full').props('label-always')

            ui.label('Select Delay Between Pictures (seconds): ').classes('mb-2 font-semibold text-center').style('font-size: 15px;') 
            delay_capture_seconds_webcam_input = ui.slider(min=0.1, max=2, value=0.5, step=0.1).classes('mb-5 w-full').props('label-always')

            with ui.row().classes('justify-between items-center w-full'):
                cancel_capture_button_webcam = ui.button('Cancel', on_click=lambda: add_persona_webcam_dialog.close(), color="gray").classes('w-1/3 m-0 text-white')
                start_capture_button_webcam = ui.button('Start', on_click=lambda: on_click_webcam_persona_add(), icon='videocam', color="blue").classes('w-1/3 m-0 text-white').style('font-size: 15px;')

        with ui.dialog().props('persistent') as add_persona_espcam_dialog, ui.card().classes('w-full sm:w-3/4 md:w-1/2 lg:w-1/2 mx-auto p-10'):
            ui.label('The application will now gather data of your face to recognise you. '
                    + 'During data collection, ensure to move your face into different poses, '
                    + 'angles and distances from the camera for best results, ensuring that there is no-one else in the frame. '
                    + 'During the process the application may hang or become unresponsive. '
                    + 'Do not exit or press any keys until the data collection process is complete.').classes('pb-2 text-justify').style('font-size: 15px;')
        
            ui.label('Select Number of Pictures: ').classes('mb-2 font-semibold text-center').style('font-size: 15px;') 
            number_pictures_espcam_input = ui.slider(min=1, max=40, value=20).classes('mb-5 w-full').props('label-always')

            ui.label('Select Delay Between Pictures (seconds): ').classes('mb-2 font-semibold text-center').style('font-size: 15px;') 
            delay_capture_seconds_espcam_input = ui.slider(min=0.1, max=2, value=0.5, step=0.1).classes('mb-5 w-full').props('label-always')

            camera_options = [0]
            if len(global_cameras) > 0:            
                camera_options = {i: camera.name for i, camera in enumerate(global_cameras)}

            ui.label('Select a Camera to Capture The Images: ').classes('mb-2 font-semibold text-center').style('font-size: 15px;') 
            camera_select_index = ui.select(camera_options, value=0).classes('w-full')

            with ui.row().classes('justify-between items-center w-full'):
                cancel_capture_button_espcam = ui.button('Cancel', on_click=lambda: add_persona_espcam_dialog.close(), color="gray").classes('w-1/3 m-0 text-white')
                start_capture_button_espcam = ui.button('Start', on_click=lambda: on_click_espcam_persona_add(), icon='videocam', color="blue").classes('w-1/3 m-0 text-white').style('font-size: 15px;')

        with ui.dialog().props('persistent') as webcam_image_capture_dialog:
            with ui.card().classes('w-full sm:w-3/4 md:w-1/2 lg:w-1/2 mx-auto p-10'):
                ui.label('Capturing and Encoding images... Please turn your attention towards the your webcam. '
                            +'Ensure that your face is well lit, approximtely 30cm from the camera, and no-one else is inside the frame.').classes('mb-2 font-semibold').style('font-size: 15px;')
                webcam_progress_bar = ui.linear_progress(value=0, show_value=False)

        with ui.dialog().props('persistent') as esp_image_capture_dialog:
            with ui.card().classes('w-full sm:w-3/4 md:w-1/2 lg:w-1/2 mx-auto p-10'):
                ui.label('Capturing and Encoding images... Please turn your attention towards the camera with the active flash. '
                            +'Ensure that your face is well lit, approximtely 30cm from the camera, and no-one else is inside the frame.').classes('mb-2 font-semibold').style('font-size: 15px;')
                espcam_progress_bar = ui.linear_progress(value=0, show_value=False)
        
        async def capture_faces_webcam(number_pictures, delay_capture_seconds):
            global global_logged_in_user, global_add_persona_thread

            encoded_faces = []

            frame_counter = 0
            progress = 0
            webcam_progress_bar.set_value(progress)

            video = cv2.VideoCapture(0)  # Open the webcam
                
            webcam_image_capture_dialog.open()

            while frame_counter < number_pictures:
                ret, frame = video.read()  # Capture frame-by-frame
                if not ret:
                    ui.notify('No WebCam Detected', color='red')
                    video.release()
    
                    webcam_image_capture_dialog.close()  
                    add_persona_webcam_dialog.close()

                    start_capture_button_webcam.enable()
                    cancel_capture_button_webcam.enable()
                    break

                faces = face_detector.findFaces(frame, draw=False)[1]

                if len(faces) > 0:
                    highest_score_face = max(faces, key=lambda face: face['score'][0], default=None)
                    if highest_score_face['score'][0] > global_application_data['face_detection_confidence_level']:
                        x, y, w, h = highest_score_face['bbox']

                        face_location = [(y, x + w, y + h, x)]
                        encoded_face = face_recognition.face_encodings(frame, face_location)

                        print('Encoding a face... ')
                        encoded_faces.append(encoded_face)
                        frame_counter += 1
                        progress = frame_counter / number_pictures
                        webcam_progress_bar.set_value(progress)

                        await asyncio.sleep(delay_capture_seconds)

            video.release()
 
            webcam_image_capture_dialog.close()  
            add_persona_webcam_dialog.close()

            start_capture_button_webcam.enable()
            cancel_capture_button_webcam.enable()

            handle_persona_add(encoded_faces)
            persona_table_ui.refresh()

        async def capture_faces_espcam(number_pictures, delay_capture_seconds, ip_address):
            global global_logged_in_user, global_add_persona_thread

            encoded_faces = []
            frame_counter = 0
            progress = 0
            espcam_progress_bar.set_value(progress)

            source = 'http://' + ip_address + ':81'
            video = cv2.VideoCapture(source)  # Open the webcam

            esp_image_capture_dialog.open()

            while frame_counter < number_pictures:
                ret, frame = video.read()  # Capture frame-by-frame
                if not ret:
                    break

                faces = face_detector.findFaces(frame, draw=False)[1]

                if len(faces) > 0:
                    highest_score_face = max(faces, key=lambda face: face['score'][0], default=None)
                    if highest_score_face['score'][0] > global_application_data['face_detection_confidence_level']:
                        x, y, w, h = highest_score_face['bbox']

                        face_location = [(y, x + w, y + h, x)]
                        encoded_face = face_recognition.face_encodings(frame, face_location)

                        print('Encoding a face... ')
                        encoded_faces.append(encoded_face)
                        frame_counter += 1
                        progress = frame_counter / number_pictures
                        espcam_progress_bar.set_value(progress)

                        await asyncio.sleep(delay_capture_seconds)  # Wait for the delay

            video.release()
    
            esp_image_capture_dialog.close()  
            add_persona_espcam_dialog.close()

            start_capture_button_espcam.enable()
            start_capture_button_espcam.enable()

            handle_persona_add(encoded_faces)
            persona_table_ui.refresh()

        def start_async_capture_faces_webcam():
            asyncio.run(capture_faces_webcam(number_pictures_webcam_input.value, delay_capture_seconds_webcam_input.value))

        def start_async_capture_faces_espcam(ip_address):
            loop = asyncio.get_event_loop()
            loop.create_task(capture_faces_espcam(number_pictures_espcam_input.value, delay_capture_seconds_espcam_input.value, ip_address))
        
        def on_click_webcam_persona_add():
            global global_add_persona_thread
            start_capture_button_webcam.disable()
            cancel_capture_button_webcam.disable()

            # Create a thread to run the asyncio loop
            global_add_persona_thread = threading.Thread(target=start_async_capture_faces_webcam, daemon=True)
            global_add_persona_thread.start()
        
        def on_click_espcam_persona_add():
            global global_cameras
            ip_address = global_cameras[camera_select_index.value].ip_address

            if not camera_is_connected_verbose(ip_address):
                return

            cancel_capture_button_espcam.disable()
            start_capture_button_espcam.disable()

            # Create a thread to run the asyncio loop
            global_add_persona_thread = threading.Thread(target=start_async_capture_faces_espcam(ip_address), daemon=True)
            global_add_persona_thread.start()

        def on_click_persona_add(registration_type):
            if registration_type == 'camera':
                if len(global_cameras) < 1:
                    ui.notify('No Cameras Added', color='red')
                    return
                add_persona_espcam_dialog.open()
                return
            elif registration_type == 'videocam':
                add_persona_webcam_dialog.open()
                return
                
        # Function to delete a persona
        def handle_persona_delete(persona_id):
            global global_personas

            global_personas = [persona for persona in global_personas if persona.id != persona_id]
                
            # Updating the store with the saved cameras
            with open(personas_path, 'wb') as f:
                pickle.dump(global_personas, f)

            preprocess_personas()
            persona_table_ui.refresh()

        @ui.refreshable
        def persona_table_ui():
            columns = [
                {'name': 'username', 'label': 'Username', 'field': 'username', 'required': True, 'align': 'left'},
                {'name': 'num_pictures', 'label': 'Number of Pictures', 'field': 'num_pictures', 'align': 'left'},
                {'name': 'date_captured', 'label': 'Date Captured', 'field': 'date_captured', 'align': 'left'},
                {'name': 'actions', 'label': 'Actions', 'align': 'right'} 
            ]

            # Create rows from global_personas list
            rows = [
                {
                    'id': persona.id,
                    'username': persona.username,
                    'num_pictures': len(persona.face_encodings),
                    'date_captured': datetime.fromtimestamp(persona.date_captured).strftime('%Y-%m-%d'),
                }
                for persona in global_personas if persona.username == global_logged_in_user.username
            ]

            # Create a table to display the global_personas
            with ui.table(columns=columns, rows=rows, row_key='username').classes('w-full mx-auto p-3') as table:
                # Add a slot for the 'actions' column to display the delete button
                table.add_slot('body-cell-actions', '''
                    <q-td key="actions" :props="props">
                        <q-btn @click="$parent.$emit('del', props)" icon="delete" flat dense />
                    </q-td>
                ''')

                table.on('del', lambda props: handle_persona_delete(props.args['row']['id']))

        persona_table_ui()

def cameras_page():
    with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
        ui.button('Menu', on_click=lambda: menu_drawer.toggle(), icon='menu').props('flat color=white')
        ui.button('Logout', on_click=lambda: logout(), icon='logout').props('flat color=white')
    
    with ui.left_drawer(bottom_corner=True).style('background-color: #d7e3f4;') as menu_drawer:
        ui.button('Dashboard', on_click=lambda: ui.navigate.to('/dashboard'), icon='dashboard').style('font-size: 15px;').props('flat color=black')
        ui.button('Personas', on_click=lambda: ui.navigate.to('/personas'), icon='face').style('font-size: 15px;').props('flat color=black')
        ui.button('Cameras', on_click=lambda: ui.navigate.to('/cameras'), icon='camera').style('font-size: 15px;').props('flat color=black')
        ui.button('Residents', on_click=lambda: ui.navigate.to('/residents'), icon='list').style('font-size: 15px;').props('flat color=black')

    with ui.card().classes('w-full mx-auto p-10') as cameras_card:
        with ui.row().classes('justify-between items-center w-full'):
            ui.label('Cameras').classes('text-3xl mb-5 text-left')
            with ui.row().classes('justify-between items-center w-0.23'):
                with ui.button(on_click=lambda: on_open_add_camera_dialog(), icon='add', color='blue').style('font-size: 15px;') as add_camera_button:
                    ui.tooltip('Add Camera').style('font-size: 15px;')

        with ui.dialog() as add_camera_dialog, ui.card().classes('w-full sm:w-3/4 md:w-1/2 lg:w-1/3 mx-auto p-10'):
            name_input = ui.input(label='Name*', validation={'Invalid characters used': lambda value: bool(re.match(r'^[A-Za-z0-9 ]*$', value))}).classes('mb-5 w-full')
            with ui.row().classes('w-full items-center'):
                ip_address_input = ui.input(label='IP Address*', validation={'Invalid characters used': lambda value: bool(re.match(r'^[0-9.]*$', value))}).classes('mb-5 w-3/4')
                ui.button(icon='wifi', on_click=lambda: camera_is_connected_verbose(ip_address_input.value)).tooltip('Test connection').classes('ml-2 w-1/6').props('flat color=gray')

            with ui.row().classes('justify-between items-center w-full'):
                ui.button('Cancel', on_click=lambda: add_camera_dialog.close(), color="gray").classes('w-1/3 m-0 text-white')
                ui.button('Register', on_click=lambda: on_add_camera_click(), color="blue").classes('w-1/3 m-0 text-white')

        def on_open_add_camera_dialog():
            if len(global_cameras) == 2:
                ui.notify(f'Only Two Cameras are Supported', color='red')
                return
            
            add_camera_dialog.open()

        def on_add_camera_click():
            camera_data = {
                'name': name_input.value.strip(),
                'ip_address': ip_address_input.value.strip(),
            }

            for camera in global_cameras:
                if camera_data['ip_address'] == camera.ip_address:
                    ui.notify(f'IP Address Taken', color='red')
                    return
                
                if camera_data['name'] == camera.name:
                    ui.notify(f'Name Taken', color='red')
                    return

            if len(camera_data['name']) == 0 or len(camera_data['ip_address']) == 0:
                ui.notify(f'Missing Required Fields', color='red')
                return
            
            handle_camera_register(camera_data)
            add_camera_dialog.close()

        def handle_camera_delete(name):
            global global_cameras

            for camera in global_cameras:
                if camera.name == name:
                    global_cameras.remove(camera)

            # Updating the store with the saved cameras
            with open(cameras_path, 'wb') as f:
                pickle.dump(global_cameras, f)

            camera_table_ui.refresh()

        def handle_camera_register(camera_data):
            global global_cameras

            new_camera = Camera(
                name=camera_data['name'],
                ip_address=camera_data['ip_address'],
            )

            global_cameras.append(new_camera)
            # Updating the store with the saved cameras
            with open(cameras_path, 'wb') as f:
                pickle.dump(global_cameras, f)

            camera_table_ui.refresh()
    
        @ui.refreshable
        def camera_table_ui():
            # Create the new table with the updated list of cameras
            columns = [
                {'name': 'name', 'label': 'Name', 'field': 'name', 'required': True, 'align': 'left'},
                {'name': 'ip_address', 'label': 'IP Address', 'field': 'ip_address', 'align': 'left'},
                {'name': 'actions', 'label': 'Actions', 'align': 'right'} 
            ]

            # Create rows for the cameras in the updated global_cameras list
            rows = [
                {
                    'name': camera.name,
                    'ip_address': camera.ip_address,
                }
                for camera in global_cameras
            ]

            # Create a table to display the global_personas
            with ui.table(columns=columns, rows=rows, row_key='name').classes('w-full mx-auto p-3') as table:
                # Add a slot for the 'actions' column to display the delete button
                table.add_slot('body-cell-actions', '''
                    <q-td key="actions" :props="props">
                        <q-btn @click="$parent.$emit('test_connection', props)" icon="wifi" flat dense/>
                        <q-btn @click="$parent.$emit('del', props)" icon="delete" flat dense/>
                    </q-td>
                ''')

                table.on('test_connection', lambda props: camera_is_connected_verbose(props.args['row']['ip_address']))
                table.on('del', lambda props: handle_camera_delete(props.args['row']['name']))
            
        camera_table_ui()

def residents_page():
    with ui.header(elevated=True).style('background-color: #3874c8').classes('items-center justify-between'):
        ui.button('Menu', on_click=lambda: menu_drawer.toggle(), icon='menu').props('flat color=white')
        ui.button('Logout', on_click=lambda: logout(), icon='logout').props('flat color=white')
    
    with ui.left_drawer(bottom_corner=True).style('background-color: #d7e3f4;') as menu_drawer:
        ui.button('Dashboard', on_click=lambda: ui.navigate.to('/dashboard'), icon='dashboard').style('font-size: 15px;').props('flat color=black')
        ui.button('Personas', on_click=lambda: ui.navigate.to('/personas'), icon='face').style('font-size: 15px;').props('flat color=black')
        ui.button('Cameras', on_click=lambda: ui.navigate.to('/cameras'), icon='camera').style('font-size: 15px;').props('flat color=black')
        ui.button('Residents', on_click=lambda: ui.navigate.to('/residents'), icon='list').style('font-size: 15px;').props('flat color=black')

    with ui.card().classes('w-full mx-auto p-10') as residents:
        def handle_user_delete(username):
            global global_personas
            global global_users
            global global_logged_in_user
            global global_application_is_running
            global global_background_application_thread

            global_personas = [persona for persona in global_personas if persona.username != username]
            preprocess_personas()

            global_users = [user for user in global_users if user.username != username]

            if global_logged_in_user.username == username:
                global_logged_in_user = None

            # Updating the store new users
            with open(users_path, 'wb') as f:
                pickle.dump(global_users, f)

            # Updating the store new personas
            with open(personas_path, 'wb') as f:
                pickle.dump(global_personas, f)

            if global_logged_in_user == None:
                global_application_is_running = False

                if global_background_application_thread:
                    global_background_application_thread.join()

                ui.navigate.to('/')
                
            residents_table_ui.refresh()
            return
        
        @ui.refreshable
        def residents_table_ui():
            ui.label('Residents').classes('text-3xl mb-5 text-center')
            columns = [
                {'name': 'username', 'label': 'Username', 'field': 'username', 'required': True, 'align': 'left'},
                {'name': 'name', 'label': 'Name', 'field': 'name', 'required': True, 'align': 'left'},
                {'name': 'preferred_lighting', 'label': 'Preferred Lighting', ':field': 'row => row.preferred_lighting == 1 ? "Dim" : row.preferred_lighting == 2 ? "Normal" : row.preferred_lighting == 3 ? "Bright" : "Unknown"', 'align': 'left'},
                {'name': 'preferred_temp', 'label': 'Preferred Temp', 'field': 'preferred_temp', 'align': 'left'},
                {'name': 'priority', 'label': 'Priority Level', 'field': 'priority', 'sortable': True, 'align': 'left'},
                {'name': 'actions', 'label': 'Actions', 'align': 'right'} 
            ]

            # Create rows from global_users list
            rows = [
                {
                    'username': user.username,
                    'name': user.name + ' ' + user.surname,
                    'preferred_lighting': user.preferred_lighting,
                    'preferred_temp': str(user.preferred_temp) + "°C",
                    'priority': user.priority,
                }
                for user in global_users
            ]
            
            # Create a table to display the residents
            table = ui.table(columns=columns, rows=rows, row_key='username').classes('w-full mx-auto p-3')

            table.add_slot('body-cell-priority', '''
                <q-td key="priority" :props="props">
                    <q-badge :color="
                        props.value == 1 ? 'green' :
                        props.value <= 3 ? 'light-green' :
                        props.value <= 5 ? 'yellow' :
                        props.value <= 7 ? 'orange' : 'red'">
                        {{ props.value }}
                    </q-badge>
                </q-td>
            ''')

            if global_logged_in_user:
                if global_logged_in_user.username == 'admin':
                    # Add a slot for the 'actions' column to display the delete button
                    table.add_slot('body-cell-actions', '''
                        <q-td key="actions" :props="props">
                            <q-btn @click="$parent.$emit('del', props)" icon="delete" flat dense/>
                        </q-td>
                    ''')
                else: 
                    # Add a slot for the 'actions' column to display the delete button
                    table.add_slot('body-cell-actions', f'''
                        <q-td key="actions" :props="props">
                            <q-btn v-if="props.row.username == '{global_logged_in_user.username}'" @click="$parent.$emit('del', props)" icon="delete" flat dense/>
                        </q-td>
                    ''')

            table.on('del', lambda props: handle_user_delete(props.args['row']['username']))

        residents_table_ui()

# ------------------ Routing for the pages ----------------------
@ui.page('/')
def index():
    if global_logged_in_user:
        ui.navigate.to('/dashboard')
        return
    login_page()

@ui.page('/register')
def register():
    if global_logged_in_user:
        ui.navigate.to('/dashboard')
        return
    registration_page()

@ui.page('/dashboard')
def dashboard():
    if not global_logged_in_user:
        ui.navigate.to('/')
        return
    dashboard_page()

@ui.page('/personas')
def personas():
    if not global_logged_in_user:
        ui.navigate.to('/')
        return
    personas_page()

@ui.page('/cameras')
def cameras():
    if not global_logged_in_user:
        ui.navigate.to('/')
        return
    cameras_page()

@ui.page('/residents')
def residents():
    if not global_logged_in_user:
        ui.navigate.to('/')
        return
    residents_page()

# Start the UI
ui.run(on_air=True)

