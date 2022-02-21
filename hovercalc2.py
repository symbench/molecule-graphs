# @Author: Michael Sandborn
# @Date:   2021-05-21T13:11:36-07:00
# @Email:  michael.sandborn@vanderbilt.edu
# @Last modified by:   michael
# @Last modified time: 2021-08-25T21:07:40-05:00


from math import radians
from multiprocessing.sharedctypes import Value
from posixpath import dirname
from shutil import move
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import csv
from csv import reader
import numpy as np
import pickle

from operator import itemgetter

DATA_DIR = os.path.join(os.getcwd(), 'data')
COMP_DIR = os.path.join(DATA_DIR, 'components')

MOTOR_DATA = os.path.join(COMP_DIR, "motors_creo.json")
PROP_DATA = os.path.join(COMP_DIR, "propellers_creo.json")
BATTERY_DATA = os.path.join(COMP_DIR, "batteries_creo.json")
CONTROLLER_DATA = os.path.join(COMP_DIR, "controllers_creo.json")

MM_PER_IN = 25.4


""" Helpers """


def loadJson(file):
    return json.load(file)


def saveJson(filename, data):
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f, indent=4)
    print(f"wrote {filename}")


def getComponentPropsWithCreoData(file):
    df = pd.read_csv(os.path.join(COMP_DIR, file))
    js = df.to_json(orient='records')
    jsd = json.loads(js)
    fname = file[:-4]+'_creo.json'
    moveComponentNameToTopOfJson(jsd, fname)


def moveComponentNameToTopOfJson(data, fname):
    size = len(data)
    names = []
    records = []
    for i in range(size):
        names.append(data[i]["Name"])
        data[i].pop("Name")
        records.append(data[i])
    with open(os.path.join(COMP_DIR, fname), 'w') as f:
        json.dump(dict(zip(names, records)), f, indent=4)
    print(f"wrote {fname}")


def pairMotorsAndProps():
    #  return a list of propellers of recommended size for each propeller
    """ can we rev eng how these pairings are made?

    """
    MOTORS = loadJson(open(MOTOR_DATA))  # dict str : properties
    PROPELLERS = loadJson(open(PROP_DATA))  # dict str : properties
    # allProps = len(set([propellers[p]['MODEL'] for p in propellers]))

    def getPropPitch(prop: str) -> float:
        return round(PROPELLERS[prop]['PITCH'] / MM_PER_IN, 2)

    def getPropDiameter(prop: str) -> float:
        return round(PROPELLERS[prop]['DIAMETER'] / MM_PER_IN, 2)

    def getMotorSizeRec(mot: str) -> tuple:
        sizeRec = MOTORS[mot]['PROP_SIZE_REC.']
        sizeRec = sizeRec.replace('(', '').replace(')', '').split(",")
        sizeRec = tuple(float(s) for s in sizeRec)
        return sizeRec

    def getMotorPitchRec(mot: str) -> tuple:
        pitchRec = MOTORS[mot]['PROP_PITCH_REC.']
        pitchRec = pitchRec.replace('(', '').replace(')', '').split(",")
        pitchRec = tuple(float(s) for s in pitchRec)
        return pitchRec

    def getRangeFromRec(rec: tuple) -> range:
        assert isinstance(rec, tuple)
        return range(int(rec[0]), int(float(rec[1]))+1)

    def getRecFromRange(rng: range) -> tuple:
        assert isinstance(rng, range)
        return (float(rng.start),float(rng.stop-1))

    # key: tuple(sizeRec, pitchRec), value: motor name
    motorPropPairs = {}
    notCovered = set()
    for motor in MOTORS:
        sizeRange = getRangeFromRec(getMotorSizeRec(motor))
        pitchRange = getRangeFromRec(getMotorPitchRec(motor))

        for propeller in PROPELLERS:
            propSize = getPropDiameter(propeller)
            propPitch = getPropPitch(propeller)

            #  add valid props for each motor
            if propSize in list(sizeRange) and propPitch in list(pitchRange):
                if motor in motorPropPairs:
                    motorPropPairs[motor].append(propeller)
                else:
                    motorPropPairs[motor] = [propeller]
            else:
                notCovered.add(propeller)

    #  now go and check that we matched correctly
    coveredProps = set()
    for motor, propList in motorPropPairs.items():
        sizeRec = getMotorSizeRec(motor)
        pitchRec = getMotorPitchRec(motor)
        print(
            f"motor: {motor} size rec: {sizeRec} pitch rec: {pitchRec}")
        sizeRange = getRangeFromRec(sizeRec)
        pitchRange = getRangeFromRec(pitchRec)  # e.g. "(5.0, 7.0)"
       
        for prop in propList:
            size = getPropDiameter(prop)
            pitch = getPropPitch(prop)
            assert size in list(sizeRange), f"invalid size {size} for size range {sizeRange}"
            assert pitch in list(pitchRange), f"invalid pitch {pitch} for pitch range {pitchRange}"
            coveredProps.add(prop)

    totalProps = len(set([item for sublist in motorPropPairs.values()
                       for item in sublist]))
    print(
        f"Done, verified {len(motorPropPairs.keys())} motors paired to {len(coveredProps)}/{totalProps} total props")

    print(f"NOT covered (no valid motor pairings): {len(notCovered)}")
    #print(list(motorPropPairs.items())[:5])

    return motorPropPairs


def pairMotorsAndBatteries():
    pass

"""
hovercalc

    This routine will, given a skeleton configuration of a vehicle consisting of
    the following: [# batteries/plates=1, # rotors=4, # vertical=4, # horizontal=0, #wings/servos=0, total Connection lengths]

- access components and creo properties in all_components.json and all_components_creo.json

output: vehicle performance, system and thruster

maybe: precompute motor and propeller pairings?

questions to answer:

- how to account for multiple batteries in hovercalc?
- how to estimate spanning area/symmetry of vehicle?

assumptions:

- no drag opposes vehicle to lift
- only vertical propellers contribute to lift

"""

# component counts
BATTERY_COUNT = 34  # 4 turnigy nanotech models don't work idx (30-33)
CONTROLLER_COUNT = 20
MOTOR_COUNT = 83
PROP_COUNT = 416  # missing from ACMs: 6x3R-RH, 5x4R-RH, 9x4_5R-RH

DESIGN_COUNT = BATTERY_COUNT * CONTROLLER_COUNT * MOTOR_COUNT * PROP_COUNT

"""
0394od_para_hub_4 -- mass:
VOLUME =  1.8750994e+04  MM^3
SURFACE AREA =  1.2010919e+04  MM^2
DENSITY =  1.3000000e-06 KILOGRAM / MM^3
MASS =  2.4376293e-02 KILOGRAM
"""
HUB_4_WT = 2.4376293e-02  # kg

"""
para_cf_fplate
VOLUME =  7.1437500e+04  MM^3
SURFACE AREA =  4.6905000e+04  MM^2
DENSITY =  1.2455957e-07 KILOGRAM / MM^3
MASS =  8.8982244e-03 KILOGRAM
"""
PLATE_WT = 8.8982244e-03  # kg

"""
0394_para_flange -- mass:
VOLUME =  1.9807498e+04  MM^3
SURFACE AREA =  9.7003345e+03  MM^2
DENSITY =  1.3000000e-06 KILOGRAM / MM^3
MASS =  2.5749748e-02 KILOGRAM
"""
FLANGE_WT = 2.5749748e-02  # kg

"""
0394OD_para_tube
VOLUME =  1.6747646e+04  MM^3
SURFACE AREA =  2.4954626e+04  MM^2
DENSITY =  1.5500747e-06 KILOGRAM / MM^3
MASS =  2.5960103e-02 KILOGRAM
Unit Length = 457.2mm == .4572m
"""
TUBE_DEFAULT_WT = 2.5960103e-02  # kg
TUBE_DEFAULT_LENGTH = .4572  # mm
TUBE_UNIT_WT = TUBE_DEFAULT_WT / TUBE_DEFAULT_LENGTH  # kg / m

"""
0394od_para_hub_tri
VOLUME =  1.5005042e+04  MM^3
SURFACE AREA =  9.6703430e+03  MM^2
DENSITY =  1.3000000e-06 KILOGRAM / MM^3
MASS =  1.9506554e-02 KILOGRAM
"""
HUB_3_TRI_WT = 1.9506554e-02  # kg


"""
Hitec_D485HW - servo for wing- how many per wing?
VOLUME =  5.7990191e+03  MM^3
SURFACE AREA =  2.3476793e+03  MM^2
DENSITY =  1.3795437e-03 KILOGRAM / MM^3
MASS =  8.0000000e+00 KILOGRAM
"""
SERVO_WT = 8  # kg - check if this is right


"""
    The class to represent a BEMP configuration.

    currently assume motors, propellers, controllers are the same for all rotors
"""


class Bemp():
    # todo add lists for vertical and horizontal motors
    def __init__(self, battery=None, controller=None, motor=None, propeller=None, rotors=4):

        # missing propellers {264, 278, 359} -- RH propellers?
        self.idxs = (battery, controller, motor, propeller)

        # todo fix the controller to the single ESC_Debugging controller
        self.battery = idxToAcm("battery", battery)
        self.controller = idxToAcm("controller", controller)
        self.motor = idxToAcm("motor", motor)
        self.propeller = idxToAcm("propeller", propeller)

        # print(componentLookup('battery', self.battery, gtoc=False))
        # print(componentLookup('controller', self.controller, gtoc=False))
        # print(componentLookup('motor', self.motor, gtoc=False))
        # print(componentLookup('propeller', self.propeller, gtoc=False))

        # number of rotors on UAV
        self.rotors = rotors

        # inp (scalars): totWeight, batteryVolts, batteryCapacity, batteryCdisRate, Kv, Kt, Rw, motorMaxCur, motorIdleCur, escMaxCur, propDia
        # prop (array of ndarrays): thrust, cthrust, torque, power, rpmLin
        self.propWt, self.inp, self.prop, self.maxrpm = retrieveComponentData(
            self.idxs, self.rotors)

        # todo make list of propellers for non-uniform configs
        self.propDiam = self.inp[-1] * 25.4  # in to mm

        self.bempWeight = self.inp[0]  # kg, weight of just BEMP

        if self.rotors == 4:  # Quad case

            self.supportLength = 1.25 * self.propDiam  # mm

            """ Compute minimum viable arm length """
            # adjacent propellers should be no closer than 1/3 prop diameter per:
            # https://aviation.stackexchange.com/questions/22269/how-much-distance-do-i-need-between-quadcopter-propellers-to-avoid-issues
            self.minArmLength = np.sqrt(.5 * (4/3) ** 2 * (self.propDiam ** 2))

            """ Compute total vehicle weight with minimal connections """

            # 1x 4-hub, 1x plate, 4x flange, 4x minArmLength tube, 4x supportLength tube
            flangeWt = 4 * FLANGE_WT
            armWt = 4 * self.minArmLength * TUBE_UNIT_WT
            supportWt = 4 * self.supportLength * TUBE_UNIT_WT

            self.vehicleWeight = self.bempWeight + \
                HUB_4_WT + flangeWt + armWt + supportWt  # kg

        elif self.rotors == 5:  # Hplane case

            self.supportLength = 1.25 * self.propDiam  # mm

            self.minArmLength = -1  # todo

            flangeWt = 5 * FLANGE_WT
            armWt = 11 * 320 * TUBE_UNIT_WT  # - Length_1 surrounding plate
            supportWt = 6 * self.supportLength * TUBE_UNIT_WT  # prop/wing support
            hubWt = HUB_4_WT + (6 * HUB_3_TRI_WT)
            servoWt = 2 * SERVO_WT
            # wingWt = 2 * RIGHT_WING_WT

            self.vehicleWeight = self.bempWeight + flangeWt + \
                armWt + supportWt + hubWt + wingWt + PLATE_WT + servoWt

        # todo
        elif self.rotors == 6:
            self.minArmLength = -1  # mm
            self.vehicleWeight = -100  # kg

    def __repr__(self):
        c = f"Bemp {self.idxs}\n"
        c += f"Rotors: {self.rotors}\n"
        c += f"Battery: {self.battery}\n"
        c += f"Controller: {self.controller}\n"
        c += f"Motor: {self.motor}\n"
        c += f"Propeller: {self.propeller}\n"
        c += f"Bemp weight: {self.bempWeight}kg\n"
        c += f"UAV weight: {self.vehicleWeight}kg\n"
        c += f"Prop Diameter: {self.propDiam}mm\n"
        c += f"Min arm length: {self.minArmLength}mm\n"
        return c


"""
    do hovercalc

    config: a list containing the following:
        [# batteries/plates=1, # rotors=4, # vertical=4, # horizontal=0, #wings/servos=0, total lengths=1(m)]


    returns: a list of component combinations that give the best hover performance for the given config

    for now assume that all motors are the same and they are all oriented in the same direction
"""


def hovercalc():

    # battery count, rotor count, vertical count, horizontal count, wing count, total length
    config = [1, 4, 4, 0, 0, 1]
    # ^ this will become input

    # component counts
    numBatteries = config[0]
    numRotors = config[1]
    numVertical = config[2]
    numHorizontal = config[3]
    numWings = config[4]  # also num servos
    totalLength = config[5]  # length of all connections in design
    assert numRotors > 0 and numVertical + numHorizontal == numRotors

    # connecting component counts
    numPlates = numBatteries // 2
    numFlanges = numRotors
    num4Hubs = numRotors // 4  # fixme

    # non BMP component mass (kg)
    nonBmpMass = (numPlates * PLATE_WT) + \
        (numFlanges * FLANGE_WT) + (num4Hubs * HUB_4_WT)

    # totalWeight = .669806616
    # print(f"total weight {totalWeight}")
    batteryVolts = input[1]
    batteryCapacity = input[2]
    batteryCdisRate = input[3]
    motorKV = input[4]
    motorKT = input[5]
    motorRW = input[6]
    motorMaxCur = input[7]
    motorIdleCur = input[8]
    escMaxCur = input[9]
    propDiameter = input[10]

    thrust = propeller[0]
    cthrust = propeller[1]
    torque = propeller[2]
    power = propeller[3]
    rpmLin = propeller[4]

    cthrust = cthrust * .9
    torque = torque * 1.1

    WF = (totalWeight * 9.81) / numv

    frictionTorque = (motorIdleCur * 10 / batteryVolts) * motorKT
    Ftorque = torque + frictionTorque  # prop torque + friction torque

    Itorque = Ftorque / motorKT  # current due to Torque

    Vtorque = (rpmLin / motorKV) + (Itorque * motorRW)  # voltage due to torque

    throttle = Vtorque / batteryVolts
    mechPower = torque * rpmLin * (2 * np.pi / 60)
    elecPower = Itorque * Vtorque
    eff = (mechPower * 100) / elecPower

    # filter
    idxFilt = []
    for i, v in enumerate(Vtorque):
        if v <= batteryVolts and Itorque[i] <= motorMaxCur:
            idxFilt.append(i)  # satisfies constraints
        else:
            pass  # ignore value

    if orientation:  # vertical

        diff = np.abs(cthrust[idxFilt] - WF)
        idxHov = np.argmin(diff)

        thrustHov = cthrust[idxHov]
        throttleHov = throttle[idxHov]
        Ihov = Itorque[idxHov]
        Vhov = Vtorque[idxHov]
        rpmHov = rpmLin[idxHov]
        PeHov = elecPower[idxHov]
        PmHov = mechPower[idxHov]
        effHov = eff[idxHov]

    else:  # horizontal - no horizontal movement at hover

        thrustHov = np.nan
        rpmHov = np.nan
        Ihov = np.nan
        PeHov = np.nan
        Vhov = np.nan
        effHov = np.nan

    # max throttle
    throttleMax, idxMaxThrottle = np.max(
        throttle[idxFilt]), np.argmax(throttle[idxFilt])

    Imax, Vmax = Itorque[idxMaxThrottle], Vtorque[idxMaxThrottle]
    rpmMax, PeMax, PmMax = rpmLin[idxMaxThrottle], elecPower[idxMaxThrottle], mechPower[idxMaxThrottle]
    effMax, thrustMax = eff[idxMaxThrottle], cthrust[idxMaxThrottle]

    # efficiency
    effOpp, idxOpp = np.max(eff[idxFilt]), np.argmax(eff[idxFilt])

    throttleOpp = throttle[idxOpp]
    Iopp, Vopp, rpmOpp = Itorque[idxOpp], Vtorque[idxOpp], rpmLin[idxOpp]
    PeOpp, PmOpp, thrustOpp = elecPower[idxOpp], mechPower[idxOpp], cthrust[idxOpp]

    # system
    Iin = np.array([Ihov, Imax, Iopp])

    thrustIn = np.array([thrustHov, thrustMax, thrustOpp])
    thrust2weight = thrustIn / WF

    # output

    totalMaxThrust = thrustMax * \
        np.array(range(1, numv+1))  # todo fix for horizontal
    # todo add multi orientation support

    # change numv to rotors
    Imotor = np.ones((rotors, len(Iin))) * Iin

    if numv > 0 and numh > 0:  # mix
        Pflight = np.array([3.65/numv, 3.65/numh])
    elif numv > 0 and numh == 0:  # vertical only
        Pflight = np.array([3.65/numv, 0])
    else:  # horizontal only
        Pflight = np.array([0, 3.65/numh])

    # fixme add support for multi orientation
    for i, o in enumerate(orientation):
        if o:  # Vertical
            Imotor = Imotor / .95 + Pflight[0] / 5  # hardcode ESC efficiency
            # print(f"vertical {i+1} Imotor {Imotor}")
        else:  # Horizontal
            Imotor = Imotor / .95 + Pflight[1] / 5
            # print(f"horizontal {i+1} Imotor {Imotor}")

    Itotal = np.sum(Imotor.T, axis=1)

    # print(f"Itotal {Itotal}")
    batCapV = (batteryCapacity * .8) / 1000
    maxC = Itotal / batCapV

    # print(f"batCapV {batCapV}")
    # print(f"Itotal {Itotal}")

    batFried = maxC > batteryCdisRate

    time = 60 / maxC

    # single thruster output -  ignore ESC fried for now
    # print("----- thruster output  -----")
    thruster = np.array([[rpmHov, rpmMax, rpmOpp], Iin, [Vhov, Vmax, Vopp], [
                        PeHov, PeMax, PeOpp], thrustIn, [effHov, effMax, effOpp], thrust2weight])

    # system output
    # print("----- system output ------")
    system = np.array(
        [np.max(totalMaxThrust), np.max(maxC), time], dtype=object)

    if True in batFried:
        return thruster, system, True
    else:
        return thruster, system, False


"""
    Read the specified file and return an ndarray of the required data

    pname: (str) the propeller data file to read from
    maxrpm: (float) the maximum rpm possible with the selected motor and battery
    propDiameter: (float) the diameter of the propeller of interest in inches
    interp: (int) the number of points to interpolate over

    returns: interpolated data for the specified propeller
"""


def parsePropFile(pname, maxrpm, propDiameter, numMotors, interp=1000):

    # todo make relative path
    pname = os.path.join(DATA_DIR, "propellers", pname)

    TABLE_LEN = 29  # number of rows in table for each RPM
    OFFSET = 4  # number of rows between RPM and start of table data

    plines = []
    with open(pname) as f:
        rd = csv.reader(f, delimiter="\t")
        for line in rd:
            plines.append(line)

    # space inconsistency in propeller dat files -__-
    sp4 = "    "
    sp5 = "     "
    sp6 = "      "
    sp7 = "       "
    sp8 = "        "

    rpms = []
    tbls = []
    for i, p in enumerate(plines):
        if p and "PROP RPM" in p[0]:
            rpm = int(p[0].split("=")[1].strip())
            rpms.append(rpm)

            # columns are V, J, Pe, Ct, Cp, Pwr, Torque, Thrust
            tbl = []  # table at RPM value
            for l in plines[i + OFFSET: i+OFFSET + TABLE_LEN+1]:
                tb = l[0].strip()
                tb = tb.replace(sp4, sp6)
                tb = tb.replace(sp5, sp6)
                tb = tb.replace(sp7, sp6)
                tb = tb.replace(sp8, sp6)
                tb = tb.replace("-NaN", "0")  # nan hacking
                tb = tb.split(sp6)

                tb = np.array([float(x) for x in tb if x != "-NaN"])
                tbl.append(tb)

            tbls.append(np.array(tbl, dtype=object))
    assert len(rpms) == len(tbls), "rpm and table len mismatch"

    rpms = np.array(rpms)

    # idxs 0-7 for cols V, J, Pe, Ct, Cp, Pwr, Torque, Thrust, respectively
    tvals = [np.array(tbls[i].T) for i in range(len(tbls))]

    # determine which prop values to use in calculations since maxRPM depends on battery voltage and motor KV
    # todo update for different motors, assume all same for now so duplicate the same value for np.array ops
    if isinstance(maxrpm, float):
        maxrpm = np.array([maxrpm for _ in range(numMotors)])

    diff = rpms - np.max(maxrpm)
    idx = np.argmin(np.abs(diff))
    if idx + 1 < len(rpms):
        rpms = rpms[:idx+2]  # to idx+1
    else:
        rpms = rpms[:idx+1]  # to idx

    """
        dictionary containing relevant prop data based on maxrpm (dependent on motor and battery)

        keys (int): rpm values for the propeller of interest
        values (np.ndarray): a list of numpy arrays indexed from 0-7 corresponding to
                cols V, J, Pe, Ct, Cp, Pwr, Torque, Thrust, respectively

    """
    d = {rpm: tvals[i] for i, rpm in enumerate(rpms)}

    # columns- V, J, Pe, Ct, Cp, Pwr, Torque, Thrust
    thrust, cthrust, torque, power = [], [], [], []
    for r in rpms:
        # thrust from raw data
        thr = d[r][7][0] * 4.44822  # N
        thrust.append(thr)

        # thrust from coefficient
        cth = d[r][3][0] * 1.225 * ((r/60)**2) * ((propDiameter*.0254) ** 4)
        cthrust.append(cth)

        # torque from raw data
        trq = d[r][6][0] * 0.11298482933333  # Nm
        torque.append(trq)

        # power from coefficient
        pwr = d[r][4][0] * 1.225 * ((r/60)**3) * \
            ((propDiameter*.0254) ** 5)  # Watts
        power.append(pwr)

    thrust, cthrust, torque, power = np.asarray(thrust), np.asarray(
        cthrust), np.asarray(torque), np.asarray(power)

    # set zero state values
    rpms = np.insert(rpms, 0, 0)
    thrust = np.insert(thrust, 0, 0)
    cthrust = np.insert(cthrust, 0, 0)
    torque = np.insert(torque, 0, 0)
    power = np.insert(power, 0, 0)

    rpmLin = np.linspace(min(rpms), max(rpms), interp)

    thrust = np.interp(rpmLin, rpms, thrust)
    cthrust = np.interp(rpmLin, rpms, cthrust)
    torque = np.interp(rpmLin, rpms, torque)
    power = np.interp(rpmLin, rpms, power)

    return thrust, cthrust, torque, power, rpmLin


""" Driver """

# if __name__ == "__main__":

#     getComponentPropsWithCreoData('propellers.csv')
# fried = {}
# for c in tqdm(configs):
#     bmp = Bemp(c[0], 0, c[1], c[2], rotors=5)
#     thruster, system, batteryFried = bmp.doHoverCalc()
#     if batteryFried:
#         fried[c] = (thruster, system)
