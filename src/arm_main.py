from dynamixel_sdk import *
import sys, tty, termios
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from collections import deque
import numpy as np
import cv2
import pyrealsense2 as rs
import time
from ultralytics import YOLO

open_maipulator = Chain(name='test_arm', links=[
    OriginLink(),
    URDFLink(
      
      name="link0",
      origin_translation=[0, 0, 0.003],
      origin_orientation=[0, 0, 0],
      rotation=[0, 1, 0],
      use_symbolic_matrix = False,
      joint_type = "revolute"
    ),     
    URDFLink(
      name="link1",
      origin_translation=[0, 0, 0.013],
      origin_orientation=[0, 0, 0],
      rotation=[0, 1, 0],
      use_symbolic_matrix = False,
      joint_type = "revolute"
    ),
    URDFLink(
      name="link2",
      origin_translation=[0.013, 0, 0],
      origin_orientation=[0, 0, 0],
      rotation=[0, 1, 0],
      use_symbolic_matrix = False,
      joint_type = "revolute"
    ),    
    URDFLink(
      name="link3",
      origin_translation=[0.013, 0, 0],
      origin_orientation=[0, 0, 0],
      rotation=[0, 1, 0],
      use_symbolic_matrix = False,
      joint_type = "revolute"
    ),
])


class OpenManipulator():
    def __init__(self, DXL_ID_list):
        self.DEVICENUM = 5
        self.DXL_MOVING_STATUS_THRESHOLD = 60
        self.DEVICENAME = '/dev/ttyUSB0'
        self.DXL_ID_list = DXL_ID_list
        self.protocol_version = 2.0 
        self.target_vector = [0.01399388  , 0.0, 0.00770011]
        self.portHandler_list = [PortHandler(self.DEVICENAME) for _ in range(self.DEVICENUM)]
        self.packetHandler_list = [PacketHandler(self.protocol_version) for _ in range(self.DEVICENUM)]

    
    def open_port_and_baud(self):
        self.TORQUE_ENABLE = 1
        self.TORQUE_DISABLE = 0
        self.ADDR_TORQUE_ENABLE = 64
        self.baudrate = 1000000
        for i in range(5):
            if self.portHandler_list[i].openPort():
                print("Succeeded to open the port | ", end="")
            else:
                print("Failed to open the port")
                
            if self.portHandler_list[i].setBaudRate(self.baudrate):
                print("Succeeded to change the baudrate | ", end="")
            else:
                print("Failed to change the baudrate")

            dxl_comm_result, dxl_error = self.packetHandler_list[i].write1ByteTxRx(self.portHandler_list[i], self.DXL_ID_list[i], self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler_list[i].getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler_list[i].getRxPacketError(dxl_error))
            else:
                print("Dynamixel has been successfully connected")

    def move_to_goal(self, dxl_goal_position):
        ADDR_GOAL_POSITION = 116
        for i in range(0, self.DEVICENUM):
            dxl_comm_result, dxl_error = self.packetHandler_list[i].write4ByteTxRx(self.portHandler_list[i], 
                                                                    self.DXL_ID_list[i], ADDR_GOAL_POSITION, dxl_goal_position[i])
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler_list[0].getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler_list[0].getRxPacketError(dxl_error))
            # [RxPacketError] Hardware error occurred. Check the error at Control Table (Hardware Error Status)! (ID: 15)
            print('rebooting.....')
            self.reboot()
    
    def kill_process(self):
        for i in range(0, self.DEVICENUM):
            dxl_comm_result, dxl_error = self.packetHandler_list[i].write1ByteTxRx(self.portHandler_list[i], self.DXL_ID_list [i], self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE)
            self.portHandler_list[i].closePort()

    def solve_ik(self):
        return  [angle_to_pos(i+180) for i in open_maipulator.inverse_kinematics(self.target_vector)*180/np.pi][1:4]
    
    def reboot(self):        
        self.packetHandler_list[-1].factoryReset(self.portHandler_list[-1], 15, 0x02)
        i = -1
        if self.portHandler_list[i].openPort():
            print("Succeeded to open the port | ", end="")
        else:
            print("Failed to open the port")
            
        if self.portHandler_list[i].setBaudRate(self.baudrate):
            print("Succeeded to change the baudrate | ", end="")
        else:
            print("Failed to change the baudrate")
        time.sleep(0.5)
        dxl_comm_result, dxl_error = self.packetHandler_list[i].write1ByteTxRx(self.portHandler_list[i], self.DXL_ID_list[i], self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler_list[i].getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler_list[i].getRxPacketError(dxl_error))
        else:
            print("Dynamixel has been successfully connected")
    

    def dump(self, dxl_goal_position, angle):
        # move to grab point
        self.target_vector[0] += 0.015
        dxl_goal_position[1:4] = self.solve_ik()
        self.move_to_goal(dxl_goal_position)
        time.sleep(1)

        # grab
        self.gripper_close(dxl_goal_position)


        # lift in z axis
        self.target_vector[2] =  0.01770011
        dxl_goal_position[1:4] = self.solve_ik()
        self.move_to_goal(dxl_goal_position)
        time.sleep(0.5)

        # move to middle
        self.target_vector[0] -= 0.01
        dxl_goal_position[1:4] = self.solve_ik()
        self.move_to_goal(dxl_goal_position)
        time.sleep(0.5)  

        # rotate in z axis
        dxl_goal_position[0] = angle_to_pos(180 + angle)
        self.move_to_goal(dxl_goal_position)
        time.sleep(0.5)

        # move to dump point
        self.target_vector[0] += 0.015
        dxl_goal_position[1:4] = self.solve_ik()
        self.move_to_goal(dxl_goal_position)
        time.sleep(1)

        # grab open
        self.gripper_open(dxl_goal_position)
        time.sleep(0.5)

        # home position
        self.target_vector = [0.01399388  , 0.0, 0.00770011]
        dxl_goal_position[1:4] = self.solve_ik()
        self.move_to_goal(dxl_goal_position)
        time.sleep(0.5)

        dxl_goal_position[0] = angle_to_pos(180)
        self.move_to_goal(dxl_goal_position)
        return dxl_goal_position
    
    def gripper_open(self, dxl_goal_position):
        dxl_goal_position[4] = angle_to_pos(100)
        self.move_to_goal(dxl_goal_position)
        time.sleep(2)

    def gripper_close(self, dxl_goal_position):
        dxl_goal_position[4] = angle_to_pos(280)
        self.move_to_goal(dxl_goal_position)
        time.sleep(2)

class realsense():
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.pipeline.start(self.config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
    
    def off(self):    
        self.pipeline.stop()
        cv2.destroyAllWindows()


def angle_to_pos(angle):
    return int(4095/360 * angle)

    
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def predict_result(model, input_image):
    predicted_results = model(input_image, imgsz=640, conf = 0.7)[0]

    detect_name = predicted_results.names

    detect_ob_num = predicted_results.boxes.cls.tolist()
    detect_ob_percentage = predicted_results.boxes.conf.tolist()
    detect_ob_cordinate = predicted_results.boxes.xyxy.tolist()

    return detect_ob_num, detect_ob_percentage, detect_ob_cordinate, detect_name


def draw_detect_object(input_image, x1, y1, x2, y2, detect_name, detect_ob_num, detect_ob_percentage, goal_point):
    cv2.circle(input_image, goal_point, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(input_image, (int((x1+x2)/2), int((y1+y2)/2)), radius=5, color=(255, 0, 0), thickness=-1)

    cv2.line(input_image, goal_point, (int((x1+x2)/2), int((y1+y2)/2)), color=(0, 0, 255), thickness=2)

    # detecting_object_text
    cv2.putText(input_image,  detect_name[detect_ob_num[0]] + " " + str(round(detect_ob_percentage[0]*100, 2)) + '%',
        (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_4) 
    
    # detecting_object
    cv2.rectangle(input_image, (x1, y1), (x2, y2), (0,255,0), 3)
    print('x : ', goal_point[0] - int((x1+x2)/2), 'y : ', goal_point[1]- int((y1+y2)/2))
    return (int((x1+x2)/2), int((y1+y2)/2))



def target_lock(deque, dx):
    NUM = 10
    deque.append(dx)
    if len(deque) > NUM:
        deque.popleft()
    if abs(sum(deque))//NUM < 10 and len(deque) == NUM:
        return True
    return False


def main():
    camera = realsense()
    openmanipulator = OpenManipulator([11, 12, 13, 14, 15])
    openmanipulator.open_port_and_baud()

    model = YOLO('./rsc/best_yolo8v_trash_x.pt')

    target_lock_arr = deque()
    target_lock_bool = False
    target_lock_name = None

    goal_point = (420, 240)
    target_point = goal_point

    dxl_goal_position = [angle_to_pos(180), *openmanipulator.solve_ik(), angle_to_pos(100)]
    openmanipulator.move_to_goal(dxl_goal_position)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))    

    while True:
        color_image = camera.get_frame()
        # predict
        detect_ob_num, detect_ob_percentage, detect_ob_cordinate, detect_name = predict_result(model, color_image)

        if detect_ob_cordinate and target_lock_bool == False:
            x1, y1, x2, y2 = map(int, detect_ob_cordinate[0])
            target_point = draw_detect_object(color_image, x1, y1, x2, y2, detect_name, detect_ob_num, detect_ob_percentage, goal_point)
            dx = target_point[0] - goal_point[0]
            # move to target
            if dx > 10:
                dxl_goal_position[0] -= angle_to_pos(0.5)
            elif dx < -10:
                dxl_goal_position[0] += angle_to_pos(0.5)

            if target_lock(target_lock_arr, dx):
                target_lock_bool = True
                target_lock_name = detect_name[detect_ob_num[0]]

            
        if target_lock_bool:
            if target_lock_name == 'pet':
                dxl_goal_position = openmanipulator.dump(dxl_goal_position, -50)            
            elif target_lock_name == 'paper':
                dxl_goal_position = openmanipulator.dump(dxl_goal_position, -90)
            elif target_lock_name == 'can':
                dxl_goal_position = openmanipulator.dump(dxl_goal_position, 50)
            elif target_lock_name == 'glass_bottle':
                dxl_goal_position = openmanipulator.dump(dxl_goal_position, 90)

            target_lock_bool = False
            target_lock_arr = deque()

        out.write(color_image)
        cv2.imshow('Color frame', color_image)
        key_input = cv2.waitKey(1)
        if key_input == 27:
            break


        dxl_goal_position[1:4] = openmanipulator.solve_ik()
        openmanipulator.move_to_goal(dxl_goal_position)
        
    openmanipulator.kill_process()
    camera.off()
    out.release()


if __name__ == '__main__':
    main()