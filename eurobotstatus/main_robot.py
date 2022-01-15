from enum import Enum

class manipulator_comand(Enum):
    GRIPPER_1_DOWN = "0x10"
    GRIPPER_1_UP = "0x11"
    GRIPPER_2_DOWN = "0x12"
    GRIPPER_2_UP = "0x13"
    GRIPPER_3_DOWN = "0x14"
    GRIPPER_3_UP = "0x15"

    PUMP_1_SUCK_ON = "0x16"
    PUMP_1_SUCK_OFF = "0x17"
    PUMP_2_SUCK_OFF = "0x18"
    PUMP_2_SUCK_ON = "0x19"
    PUMP_3_SUCK_OFF = "0x1A"
    PUMP_3_SUCK_ON = "0x1B"

    # IF WE LOOK FROM TOP TO BOTTOM THE GRIPPER'S LOCATION IS IN FRONT OF THE ROBOT'S BODY
    # SCHEME:
    #        ()(L)()1()(R)()
    #         ()         ()
    #          ()       ()
    #           ()     ()
    #            ()   ()
    #              ()()
    #

    VALVE_1_LEFT_OPEN = "0x20"
    VALVE_1_LEFT_CLOSE = "0x21"
    VALVE_1_RIGHT_OPEN = "0x22"
    VALVE_1_RIGHT_CLOSE = "0x23"
    VALVE_2_LEFT_OPEN = "0x24"
    VALVE_2_LEFT_CLOSE = "0x25"
    VALVE_2_RIGHT_OPEN = "0x26"
    VALVE_2_RIGHT_CLOSE = "0x27"
    VALVE_3_LEFT_OPEN = "0x28"
    VALVE_3_LEFT_CLOSE = "0x29"
    VALVE_3_RIGHT_OPEN = "0x2A"
    VALVE_3_RIGHT_CLOSE = "0x2B"