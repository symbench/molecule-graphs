# @Author: Michael Sandborn
# @Date:   2021-05-21T13:11:36-07:00
# @Email:  michael.sandborn@vanderbilt.edu
# @Last modified by:   michael
# @Last modified time: 2021-08-25T21:07:40-05:00


#
# Created on Tue Feb 01 2022
#
# Copyright (c) 2022 Your Company
#

import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import csv
from csv import reader
import numpy as np
import zipfile
import pickle
import _pickle as cPickle
import time
import random

import bz2

from operator import itemgetter



"""
todo on hovercalc

- access components in all_components.json
- set default length for each connection, and shrink as components evolve
- use graph coloring for component types, propeller orientation
- pass a list of component types and quantities

input: [for each vehicle, provide vectors specifying [number of rotors, number of batteries, orientation of each rotor]]

output: vehicle performance, system and thruster

maybe: precompute motor and propeller pairings?

questions to answer:

- how to account for multiple batteries in hovercalc?
- how to estimate spanning area of vehicle?
- how to get convex hull of vehicle by node position to estimate symmetry?
- how to introduce fitness function term which removes subgraphs to choose active nodes?

"""

# Assumes a parent directory containing corpus
CDIR = os.path.join(os.pardir, "athens-uav-corpus/CorpusSpreadsheets")
CMP_GRAPH_DIR = os.path.join(os.pardir, "athens-uav-corpus/OpenMETAModels/components")

# component counts
BATTERY_COUNT = 34 # 4 turnigy nanotech models don't work idx (30-33)
CONTROLLER_COUNT = 19
MOTOR_COUNT = 83
PROP_COUNT = 417 # missing from ACMs: 6x3R-RH, 5x4R-RH, 9x4_5R-RH

DESIGN_COUNT = BATTERY_COUNT * CONTROLLER_COUNT * MOTOR_COUNT * PROP_COUNT

HOVER_DIR = os.path.join(os.pardir, "hovercalc-copy")
CORPUS_DIR = os.path.join(os.pardir, "athens-uav-corpus", "CorpusSpreadsheets")
INP_FILE = os.path.join(HOVER_DIR, "hovCalcInputCopy.csv")
PROP_DIR = os.path.join(HOVER_DIR, "HoverCalcExecution", "PropData")

NUM_MOTORS = 4

# for excel data handling
OFFSET = 4
NUM_RUNS = 100

jfiles = [x for x in os.listdir('data/') if ".json" in x]
COMPONENTS = ['battery', 'controller', 'motor', 'propeller']

"""

0394od_para_hub_4 -- mass:
VOLUME =  1.8750994e+04  MM^3
SURFACE AREA =  1.2010919e+04  MM^2
DENSITY =  1.3000000e-06 KILOGRAM / MM^3
MASS =  2.4376293e-02 KILOGRAM

para_cf_fplate
VOLUME =  7.1437500e+04  MM^3
SURFACE AREA =  4.6905000e+04  MM^2
DENSITY =  1.2455957e-07 KILOGRAM / MM^3
MASS =  8.8982244e-03 KILOGRAM

0394_para_flange -- mass:
VOLUME =  1.9807498e+04  MM^3
SURFACE AREA =  9.7003345e+03  MM^2
DENSITY =  1.3000000e-06 KILOGRAM / MM^3
MASS =  2.5749748e-02 KILOGRAM

0394OD_para_tube
VOLUME =  1.6747646e+04  MM^3
SURFACE AREA =  2.4954626e+04  MM^2
DENSITY =  1.5500747e-06 KILOGRAM / MM^3
MASS =  2.5960103e-02 KILOGRAM
Unit Length = 457.2mm == .4572m

para_wing_right
VOLUME =  4.4313726e+06  MM^3
SURFACE AREA =  3.3668448e+05  MM^2
DENSITY =  2.7679900e-08 KILOGRAM / MM^3
MASS =  1.2265995e-01 KILOGRAM

0394od_para_hub_tri
VOLUME =  1.5005042e+04  MM^3
SURFACE AREA =  9.6703430e+03  MM^2
DENSITY =  1.3000000e-06 KILOGRAM / MM^3
MASS =  1.9506554e-02 KILOGRAM

Hitec_D485HW
VOLUME =  5.7990191e+03  MM^3
SURFACE AREA =  2.3476793e+03  MM^2
DENSITY =  1.3795437e-03 KILOGRAM / MM^3
MASS =  8.0000000e+00 KILOGRAM
"""

HUB_3_TRI_WT = 1.9506554e-02 # kg
HUB_4_WT = 2.4376293e-02 # kg
PLATE_WT = 8.8982244e-03 # kg
FLANGE_WT = 2.5749748e-02 # kg
TUBE_DEFAULT_WT = 2.5960103e-02 # kg
TUBE_DEFAULT_LENGTH = 457.2 # mm
TUBE_UNIT_WT = TUBE_DEFAULT_WT / TUBE_DEFAULT_LENGTH # kg / mm

RIGHT_WING_WT = 1.2265995e-01 # kg

# from datasheet, Creo says this is 8kg?
SERVO_WT = 4.5e-2 # kg


"""
    Flight Dynamics Parser:

    Parses flight dynamics input and output files for the flight path visualizer
"""
class FlightDynamicsReport():

    def __init__(self, fname):
        self.filename =  fname

        timeSeries = self._parseTimeSeries()

        if timeSeries is not None:
            self.time = timeSeries[0]
            self.phi = timeSeries[1]
            self.theta = timeSeries[2]
            self.psi = timeSeries[3]

            self.issue = False
        else:
            self.issue = True

    def __repr__(self):
        return f"File: {self.filename} \nIssue: {self.issue}"


    def _parseTrimStates(self):
        fname = self.filename
        with open(fname) as f:
            data = f.readlines()

        # return trim state velocities, pqr, xyz etc

        # num trim states, speeds, xyz, pqrf

    def _parseTimeSeries(self):
        fname = self.filename
        with open(fname) as f:
            timeSrs = f.readlines()
            for i, t in enumerate(timeSrs):
                if "time       phi      theta" in t:
                    #print(t)
                    TABLE_START = i
                if "Calculation completed at time" in t:
                    #print(t)
                    TABLE_END = i - 1

            # time, phi, theta, psi 0-3
            table = []
            for data in timeSrs[TABLE_START +2:TABLE_END]:
                data = data.strip()
                data = data.replace("    ",  "     ")
                data = data.replace("   ",  "     ")
                data = data.replace("  ",  "     ")
                data = data.split("     ")
                while  '' in data:
                    data.remove('')

                row = []
                for d in data:
                    row.append(float(d.strip()))
                table.append(np.array(row))

            if not table:
                return None
            else:
                table = np.array(table)
                time = table[:, 0]
                phi = table[:, 1]
                theta = table[:, 2]
                psi = table[:, 3]

            return (time, phi, theta, psi)


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
        self.propWt, self.inp, self.prop, self.maxrpm = retrieveComponentData(self.idxs, self.rotors)

        # todo make list of propellers for non-uniform configs
        self.propDiam = self.inp[-1] * 25.4 # in to mm

        self.bempWeight = self.inp[0] # kg, weight of just BEMP

        if self.rotors == 4: # Quad case

            self.supportLength = 1.25 * self.propDiam # mm

            """ Compute minimum viable arm length """
            # adjacent propellers should be no closer than 1/3 prop diameter per:
            # https://aviation.stackexchange.com/questions/22269/how-much-distance-do-i-need-between-quadcopter-propellers-to-avoid-issues
            self.minArmLength = np.sqrt(.5 * (4/3) ** 2 * (self.propDiam ** 2))

            """ Compute total vehicle weight with minimal connections """

            # 1x 4-hub, 1x plate, 4x flange, 4x minArmLength tube, 4x supportLength tube
            flangeWt = 4 * FLANGE_WT
            armWt = 4 * self.minArmLength * TUBE_UNIT_WT
            supportWt = 4 * self.supportLength * TUBE_UNIT_WT

            self.vehicleWeight = self.bempWeight + HUB_4_WT + flangeWt + armWt + supportWt # kg

        elif self.rotors == 5: # Hplane case

            self.supportLength = 1.25 * self.propDiam # mm

            self.minArmLength = -1 # todo

            flangeWt = 5 * FLANGE_WT
            armWt = 11 * 320 * TUBE_UNIT_WT #  - Length_1 surrounding plate
            supportWt = 6 * self.supportLength * TUBE_UNIT_WT # prop/wing support
            hubWt = HUB_4_WT + (6 * HUB_3_TRI_WT)
            servoWt = 2 * SERVO_WT
            wingWt = 2 * RIGHT_WING_WT

            self.vehicleWeight = self.bempWeight + flangeWt + armWt + supportWt + hubWt + wingWt + PLATE_WT + servoWt

        # todo
        elif self.rotors == 6:
            self.minArmLength = -1 # mm
            self.vehicleWeight = -100 # kg

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

    def doHoverCalc(self):
        # todo where to write output
        return hoverCalcNumpy(self)


def bempIdxsToNames(idxs):
    battery = idxToName("battery", idxs[0])
    controller = idxToName("controller", idxs[1])
    motor = idxToName("motor", idxs[2])
    propeller = idxToName("propeller", idxs[3])
    return [battery, controller, motor, propeller]

def bempIdxToAcms(idxs):
    battery = idxToAcm("battery", idxs[0])
    controller = idxToAcm("controller", idxs[1])
    motor = idxToAcm("motor", idxs[2])
    propeller = idxToAcm("propeller", idxs[3])
    return [battery, controller, motor, propeller]

"""
    do hovercalc

    bmp (Bemp) : a Bemp object representing a single configuration to be simulated.

    returns: (list) the results of the hovercacl simulation from the received configuration


    for now assume that all motors are the same and they are all oriented in the same direction
"""
def hoverCalcNumpy(bmp):

# todo orientation, num motors, vertical horizontal, add support for different motors and propellers

    # input: totWeight, batteryVolts, batteryCapacity, batteryCdisRate, Kv, Kt, Rw, motorMaxCur, motorIdleCur, escMaxCur, propDia
    # propeller: thrust, cthrust, torque, power, rpmLin
    input, propeller, maxrpm, rotors = bmp.inp, bmp.prop, bmp.maxrpm, bmp.rotors

    if rotors == 4: # quadcopter
        numv = rotors
        numh = 0 # all vertical for now
        orientation = [1,1,1,1] # 1 is vertical, 0 is horizontal
    elif rotors == 5: # Hplane
        numv = rotors - 1
        numh = 1
        orientation = [1,1,1,1,0]

    #totalWeight = input[0]
    # todo
    totalWeight = bmp.vehicleWeight

    #totalWeight = .669806616
    #print(f"total weight {totalWeight}")
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
    Ftorque = torque + frictionTorque # prop torque + friction torque

    Itorque = Ftorque / motorKT # current due to Torque

    Vtorque = (rpmLin / motorKV) + (Itorque * motorRW) # voltage due to torque

    throttle = Vtorque / batteryVolts
    mechPower = torque * rpmLin * (2 * np.pi / 60)
    elecPower = Itorque * Vtorque
    eff = (mechPower * 100) / elecPower

    # filter
    idxFilt = []
    for i, v in enumerate(Vtorque):
        if v <= batteryVolts and Itorque[i] <= motorMaxCur:
            idxFilt.append(i) # satisfies constraints
        else:
            pass #ignore value

    if orientation: # vertical

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

    else: # horizontal - no horizontal movement at hover

        thrustHov = np.nan
        rpmHov = np.nan
        Ihov = np.nan
        PeHov = np.nan
        Vhov = np.nan
        effHov = np.nan

    ## max throttle
    throttleMax, idxMaxThrottle = np.max(throttle[idxFilt]), np.argmax(throttle[idxFilt])

    Imax, Vmax = Itorque[idxMaxThrottle], Vtorque[idxMaxThrottle]
    rpmMax, PeMax, PmMax = rpmLin[idxMaxThrottle], elecPower[idxMaxThrottle], mechPower[idxMaxThrottle]
    effMax, thrustMax = eff[idxMaxThrottle], cthrust[idxMaxThrottle]

    ## efficiency
    effOpp, idxOpp = np.max(eff[idxFilt]), np.argmax(eff[idxFilt])

    throttleOpp = throttle[idxOpp]
    Iopp, Vopp, rpmOpp = Itorque[idxOpp], Vtorque[idxOpp], rpmLin[idxOpp]
    PeOpp, PmOpp, thrustOpp = elecPower[idxOpp], mechPower[idxOpp], cthrust[idxOpp]

    ## system
    Iin = np.array([Ihov, Imax, Iopp])

    thrustIn = np.array([thrustHov, thrustMax, thrustOpp])
    thrust2weight = thrustIn / WF

    ## output

    totalMaxThrust = thrustMax * np.array(range(1, numv+1)) # todo fix for horizontal
    # todo add multi orientation support

    # change numv to rotors
    Imotor = np.ones((rotors, len(Iin))) * Iin

    if numv > 0 and numh > 0: # mix
        Pflight = np.array([3.65/numv, 3.65/numh])
    elif numv > 0 and numh == 0: # vertical only
        Pflight = np.array([3.65/numv, 0])
    else: # horizontal only
        Pflight = np.array([0, 3.65/numh])

    # fixme add support for multi orientation
    for i, o in enumerate(orientation):
        if o: # Vertical
            Imotor = Imotor / .95 + Pflight[0] / 5 # hardcode ESC efficiency
            #print(f"vertical {i+1} Imotor {Imotor}")
        else: # Horizontal
            Imotor = Imotor / .95 + Pflight[1] / 5
            #print(f"horizontal {i+1} Imotor {Imotor}")

    Itotal = np.sum(Imotor.T, axis=1)

    #print(f"Itotal {Itotal}")
    batCapV = (batteryCapacity * .8) / 1000
    maxC = Itotal / batCapV

    #print(f"batCapV {batCapV}")
    #print(f"Itotal {Itotal}")

    batFried = maxC > batteryCdisRate

    time = 60 / maxC

    # single thruster output -  ignore ESC fried for now
    #print("----- thruster output  -----")
    thruster = np.array([[rpmHov, rpmMax, rpmOpp], Iin, [Vhov, Vmax, Vopp], [PeHov, PeMax, PeOpp], thrustIn, [effHov, effMax, effOpp], thrust2weight])

    # system output
    #print("----- system output ------")
    system = np.array([np.max(totalMaxThrust), np.max(maxC), time], dtype=object)

    if True in batFried:
        return thruster, system, True
    else:
        return thruster, system, False


"""
    fname: the name of the file to read
    cols: the columns of data to extract
    avg: if True, will return the average of each column

    returns pandas dataframe containing the selected columns
"""
def readComponentSheet(fname, cols, avg=False):
    f = os.path.join(CDIR, fname)
    df = pd.read_excel(f, engine='openpyxl')

    df = df[cols]

    if avg:
        av = df.mean(axis=0)
        return df, av
    return df

"""
    read the component data from the excel spreadsheet and save it in a
    json file named after the component type. Focus on the battery, propeller, motor, and controller

    cmp (str): the component to read data from. this string determines which columns to load from the
               component spreadsheets
"""
def getComponentData(cmp='battery', avg=False):
    if cmp == 'battery':
        # component-specific columns
        cols = ['Name',
                'Cost [$]',
                'Length [mm]',
                'Width [mm]',
                'Thickness [mm]',
                'Weight [g]',
                'Voltage [V]',
                'Number of Cells',
                'Chemistry Type',
                'Capacity [mAh]',
                'Peak Discharge Rate [C]',
                'Cont. Discharge Rate [C]'
                ]

        fname = "Battery_Corpus.xlsx"
        if avg:
            data, avg = readComponentSheet(fname, cols, avg)
            print(f"[{cmp}] - done with {data.shape[0]} rows")
            print(f"Average battery: {average}")
        else:
            data = readComponentSheet(fname, cols, avg)

        # FIXME - Turnigy NanoTech batteries don't work in graph
        # so drop these rows for now
        data = data[:30]

        data.dropna(inplace=True)
        data.to_json('batteries.json', orient='split')
        print(f"[ {cmp} ] - done with {data.shape[0]} rows")

    elif cmp == 'motor':

        cols = ['Name',
                'Cost [$]',
                'Total Length [mm]',
                'Can Diameter [mm]',
                'Weight [g]',
                'KV [RPM/V]',
                'KT [Nm/A]',
                'KM [Nm/sqrt(W)]',
                'Max Current [A]',
                'Max Power [W]',
                'Internal Resistance [mOhm]',
                'Io Idle Current@10V [A]',
                'Poles'
                ]

        fname = "Motor_Corpus.xlsx"

        if avg:
            data, avg = readComponentSheet(fname, cols, avg)
            print(f"[{cmp}] - done with {data.shape[0]} rows")
            print(f"Average {cmp}: {average}")
        else:
            data = readComponentSheet(fname, cols, avg)

        data.dropna(inplace=True)
        data.to_json('motors.json', orient='split')

        print(f"[ {cmp} ] - done with {data.shape[0]} rows")

    elif cmp == 'propeller':
        cols = ['Name',
                'Cost ($)',
                'Hub Diameter [mm]',
                'Hub Thickness [mm]',
                'Shaft Diameter [mm]',
                'Diameter [mm]',
                'Pitch [mm]',
                'Weight [grams]',
                'Performance File'
                ]

        fname = "Propeller_Corpus_Rev3.xlsx"

        if avg:
            data, avg = readComponentSheet(fname, cols, avg)
            print(f"[{cmp}] - done with {data.shape[0]} rows")
            print(f"Average {cmp}: {average}")
        else:
            data = readComponentSheet(fname, cols, avg)

        data.dropna(inplace=True)
        data.to_json('propellers.json', orient='split')

        print(f"[ {cmp} ] - done with {data.shape[0]} rows")

    elif cmp == 'controller':

        cols = ['Name',
                'Cost [$]',
                'Length [mm]',
                'Width [mm]',
                'Thickness [mm]',
                'Weight [g]',
                'Max Voltage [V]',
                'Peak Amps [A]',
                'Cont Amps [A]'
                ]

        fname = "Controls_Corpus.xlsx"

        if avg:
            data, avg = readComponentSheet(fname, cols, avg)
            print(f"[{cmp}] - done with {data.shape[0]} rows")
            print(f"Average {cmp}: {average}")
        else:
            data = readComponentSheet(fname, cols, avg)

        data.dropna(inplace=True)
        data.to_json('controllers.json', orient='split')

        print(f"[ {cmp} ] - done with {data.shape[0]} rows")

"""
    Load a specific instance of a component from the json data extracted from the component
    spreadsheets.

    idx: (int > -1) the index of the component of interest
    name: (str) - TODO: add load by name functionality
    cmp: (str) - the component whose information to load

    returns: (dict) - { property name : property value } for the selected component
"""
def readComponentProps(idxs):

    cmp = ['battery', 'controller', 'motor', 'propeller']
    assert len(idxs) == len(cmp), "idx and cmp length must match"

    properties = []
    for i, idx in enumerate(idxs):

        if cmp[i] == 'battery':
            ckeys = ['name', 'cost', 'length', 'width',
                    'thickness', 'weight', 'voltage',
                    'numCells', 'chemType', 'capacity', 'pDis', 'cDis']
        elif cmp[i] == 'controller':
            ckeys = ['name', 'length', 'width', 'thickness', 'weight', 'peakAmps', 'contAmps']
        elif cmp[i] == 'motor':
            ckeys = ['name', 'cost', 'length', 'diameter',
                     'weight', 'KV', 'KT', 'KM', 'maxCurrent', 'maxPower', 'intRes', 'idleCur', 'poles']
        elif cmp[i] == 'propeller':
            ckeys = ['name', 'cost', 'hubDiameter', 'hubThickness',
                     'shaftDiameter', 'diameter', 'pitch', 'weight',
                     'performanceFile']

        fname = cmp[i] + ".json"
        assert fname in jfiles, "file not found!"

        with open("data/"+fname) as f:
            cmps = json.load(f)

        data = cmps['data']
        assert idx <= cmps['index'][-1], "invalid component index"

        for j, cvals in enumerate(data):
            if idx == j: # found the componet, return its properties
                properties.append(dict(zip(ckeys, cvals)))

    assert len(properties) == len(cmp), "data fetched improperly"
    return properties


"""
    Load the json file for the requested component and
    return the columns giving the index and name of each
    component instance
"""
def loadNameIdx(cmp, namesOnly=False):
    assert cmp in COMPONENTS, "invalid component"

    fname = cmp + ".json"
    assert fname in jfiles, "file not found!"

    with open("data/" + fname) as f:
        cmps = json.load(f)
        idx = cmps['index']
        names = [x[0].strip() for x in cmps['data']]

    # used by resolveComponentCorpusGraphNames
    if namesOnly:
        return names
    return idx, names


"""
    Write the names of each component type along with their indices
    to a json file in dictionary format

    nameToIdx --> names are keys, idx are values
    idxToName --> idxs are names, keys are values
"""
def writeComponentNameIdx(cmp='battery', nameToIdx=True):

    idx, name = loadNameIdx(cmp)
    print(f'loading {len(list(zip(idx, name)))}')

    if nameToIdx:
        d = dict(zip(name, idx))
        fn = 'nameToIdx'
    else:
        d = dict(zip(idx, name))
        fn = 'idxToName'

    # json dump the resulting dictionary
    fname = cmp + "_" + fn + ".json"
    with open(fname, 'w') as f:
        json.dump(d, f)
    print(f"created {fname}")

"""
    Lookup the index of a component using its name
"""
def nameToIdx(cmp, name):
    fname = os.path.join("data/", cmp + "_nameToIdx.json")
    with open(fname) as f:
        d = json.load(f)
    return d[name]

"""
    Lookup the name of a component using its index
"""
def idxToName(cmp, idx):
    fname = os.path.join("data/", cmp + "_idxToName.json")
    with open(fname) as f:
        d = json.load(f)
    if not isinstance(idx, str):
        idx = str(idx)
    return d[idx]

"""
    Lookup the ACM name of a component using its index
"""
def idxToAcm(cmp, idx, reverse=False):
    fname = os.path.join("data/", cmp + "_idxToAcm.json")
    with open(fname) as f:
        d = json.load(f)
    if not isinstance(idx, str):
        idx = str(idx)
    if reverse:
        d = {v : k for k, v in d.items()}
    return d[idx]



"""
    Get the count for each of the 4 component types
"""
def getComponentCounts():
    f = ['battery', 'controller', 'motor', 'propeller']
    counts = []
    for fi in f:
        fn = fi + ".json"
        with open(fn) as fo:
            d = json.load(fo)
            counts.append(d['index'][-1])
    # battery, controller, motor, propeller
    return counts

"""
    Update component json files from corpora spreadsheets
"""
def updateComponent(cmp='battery'):
    getComponentData(cmp, False)


"""
    Read the graph names from the CAD / components folder
"""
def extractGraphComponentNames():
    fnames = []
    for i, (r,s,f) in enumerate(os.walk(CMP_GRAPH_DIR)):

        if f and "flange" not in f[0] and "tube" not in f[0] and "plate" not in f[0]:
            #print(f[0])
            fnames.append(f[0])
    print(f"{len(fnames)} graph components loaded")
    return fnames

"""
    Read the component names from the corpus spreadsheets folder
"""
def extractCorpusComponentNames():
    cmps = ['battery', 'controller', 'motor', 'propeller']
    cnames = []
    for cmp in cmps:
        names = loadNameIdx(cmp, namesOnly=True)
        for name in names:
            #print(name)
            cnames.append(name)
    print(f"{len(cnames)} corpus components loaded")
    return cnames

"""
    Write the file that allows the lookups for component names
"""
def resolveComponentCorpusGraphNames():

    gnames = set(extractGraphComponentNames())
    cnames = set(extractCorpusComponentNames())

    print(gnames)

    # graph components
    bats = []
    prps = []
    trls = []
    mots = []
    for g in gnames:
        if "Turnigy" in g:
            #print(f"adding {g} to batteries")
            bats.append(g)
        elif "apc_propellers" in g:
            #print(f"adding {g} to propellers")
            prps.append(g)
        elif g[-1] == "A" or "FLAME" in g or "AIR" in g or "ALPHA" in g or "AT " in g or "UAS35" in g or "UAS55" in g or "UAS20LV" in g or "T " in g:
            print(f"adding {g} to controllers")
            trls.append(g)
        else:
            #print(f"adding {g} to motors")
            mots.append(g)

    # corpus components
    cbats = []
    cprps = []
    ctrls = []
    cmots = []
    for c in cnames:
        if "Turnigy" in c:
            #print(f"adding {c} to cbatteries")
            cbats.append(c)
        elif "x" in c:
            #print(f"adding {c} to cpropellers")
            cprps.append(c)
        elif c[-1] == "A" or "FLAME" in c or "AIR" in c or "ALPHA" in c or "AT " in c or "UAS35" in c or "UAS55" in c or "UAS20LV" in c or "T " in c:
        #elif c[-1] == "A" or c[-1] == " A":
            #print(f"adding {c} to ccontrollers")
            ctrls.append(c)
        else:
            #print(f"adding {c} to cmotors")
            cmots.append(c)

    graphNameToCorpusName = {}
    graphNameToCorpusName['battery'] = {}
    graphNameToCorpusName['controller'] = {}
    graphNameToCorpusName['propeller'] = {}
    graphNameToCorpusName['motor'] = {}

    # batteries
    batteries = list(zip(sorted(bats), sorted(cbats)))
    for b in batteries:
        graphNameToCorpusName['battery'][b[0]] = b[1]
    print(f" {len(graphNameToCorpusName['battery'].keys())} --- size after batteries  ")

    # props
    for c in cprps:
        #print(c)
        for p in prps:

            # fixed name structure
            # graph name: apc_propellers_9x3.8SF.acm
            # component name: 9x3.8SF
            sidx = p.index("s_") + 2
            eidx = p.index(".acm")
            pn = p[sidx : eidx]
            #print(pn, c)
            if c == pn:
                graphNameToCorpusName['propeller'][p] = c
                continue
            elif c == pn + " ":
                graphNameToCorpusName['propeller'][p] = c
                continue
    print(f" {len(graphNameToCorpusName['propeller'].keys())} --- size after propellers  ")

    # controllers
    for c in ctrls:
        for t in trls:
            if c in t:
                graphNameToCorpusName['controller'][t] = c
                continue
    print(f" {len(graphNameToCorpusName['controller'].keys())} --- size after controllers  ")

    # motors
    for m in cmots:
        for o in mots:
            if m in o:
                graphNameToCorpusName['motor'][o] = m
                continue
    print(f" {len(graphNameToCorpusName['motor'].keys())} --- size after motors  ")

    print(graphNameToCorpusName)

    with open('graphToCorpusNames.json', 'w') as gtc:
         json.dump(graphNameToCorpusName, gtc)
    print("done")


"""
    Get the requested name of a component

    gtoc: boolean indicating the whether the name is a graph component or a corpus component

    gtoc = True
        --> input: graph component name
        --> output: corpus component name

    gtoc = False
        --> input: corpus component name
        --> output: graph component name

    cmp: str - one of the following: 'battery', 'controller', 'motor', 'propeller' to indicate which component names to search
    name: str - the name of the component to resolve

    return: either the component's corpus name or graph name, depending on gtoc

"""
def componentLookup(cmp, name, gtoc=True, graph=False):

    # assume we want to map from graph to corpus names
    with open('graphToCorpusNamesCopy.json') as gtc:
        gtocNames = json.load(gtc)

    names = gtocNames[cmp]

    if gtoc:
        return names[name]

    #  graph names given corpus names
    names = {v: k for (k, v) in names.items()}

    if graph:
        return names

    print(len(names.items()))
    return names[name]


"""
    Look up the ACM graph name of a component from its corpus name

    the following props are missing from ACMs: 6x3R-RH, 5x4R-RH, 9x4_5R-RH
"""
def acmLookup(cname, ctog=True):
    with open('corpusNametoAcmName.json') as g:
        cnames = json.load(g)

    if not ctog:
        # graph name to corpus name
        cnames = {v: k for (k, v) in cnames.items()}

    return cnames[cname]


"""
    retrieve the component properties for a single BEMP configuration and resolve the names
    from the component names to the graph component names

    idxs: (tup) - a tuple of the indices to lookup properties for each of the BEMP components
    rotors: (int) - the number of rotors present on the vehicle (default is 4)
"""
def retrieveComponentData(idxs, numRotors):

    # battery
    bat, con, mot, prop = readComponentProps(idxs)

    # battery
    batteryVolts = float(bat['voltage'])
    batteryCapacity = float(bat['capacity'])
    batteryCdisRate = float(bat['cDis'])
    bwt = float(bat['weight'])

    # controller
    escMaxCur = float(con['contAmps']) # max continuous current
    cwt = numRotors * float(con['weight']) # 4 controllers

    # motor
    Kv = float(mot['KV'])
    Kt = float(mot['KT'])
    Rw = float(mot['intRes']) / 1000 # convert mOhms to Ohms
    motorMaxCur = float(mot['maxCurrent'])
    motorIdleCur = float(mot['idleCur'])
    mwt = numRotors * float(mot['weight']) # 4 motors

    # propeller
    propDia = float(prop['diameter']) / 25.4 # to inches
    propf = prop['performanceFile']
    propWt = float(prop['weight'])
    pwt = numRotors * propWt

    # 1 battery, 4 controller, 4 motors, 4 propellers
    totWeight = (bwt + cwt + mwt + pwt) / 1000 # to Kg

    maxrpm = Kv * batteryVolts # motor and battery combo dictates max rotor rpm
    #print(f"maxrpm is {maxrpm}")

    thrust, cthrust, torque, power, rpmLin = parsePropFile(propf, maxrpm, propDia)

    inp = np.array([totWeight, batteryVolts, batteryCapacity, batteryCdisRate, Kv, Kt, Rw, motorMaxCur, motorIdleCur, escMaxCur, propDia])

    prop = np.array([thrust, cthrust, torque, power, rpmLin])

    return propWt, inp, prop, maxrpm

"""
    call processResults to get cleaned data and create visualizations e.g. radar chart viz and distributions of certain attributes e.g. mass, total thrust etc. -- return these assets to simulyzer to be displayed in the UI
"""
def getResults(inFile, outFile):

    with open(inFile, 'r') as i:
        inputs = []
        ci = reader(i)
        for r in ci:
            inputs.append(r)

    with open(outFile, 'r') as f:
        outputs = []
        co = reader(f)
        for r in co:
            outputs.append(r)

    #print(len(inputs))
    #print(len(outputs))
    #assert len(inputs) == len(outputs), "length mismatch"

    cleanOutputs = []
    cleanInputs = []
    idxs = []
    lens = set()
    for i, data in enumerate(outputs):
        #print(data)
        entry = data[0].replace('"', "")
        #print(entry)

        if "t-motor" in entry[:50]:
            #print("t-motor")
            batNameEnd = entry.index("t-motor") - 1
        elif "kde_direct" in entry[:50]:
            #print("kde_direct")
            batNameEnd = entry.index("kde") - 1

        batName = entry[:batNameEnd]
        rem = entry[batNameEnd:]
        rem = rem.split()
        rem.insert(0, batName)
        #print(rem)
        lens.add(len(rem))
        cleanOutputs.append(rem)

    return inputs, cleanOutputs

"""
    Remove the bad runs i.e. where battery was fried
"""
def filterResults(inputFile, outputFile):

    inputs, outputs = getResults(inputFile, outputFile)

    filteredOutputs = []

    idxs = []
    obad = 0
    # keep output data from runs where the battery is not fried
    for i, o in enumerate(outputs):
        if o[o.index("Horizontal")-1] == "0":
            filteredOutputs.append(o)
            idxs.append(i)
        else:
            obad +=1

    # keep input data from runs where the battery is not fried
    filteredInputs = [inputs[i] for i in idxs]

    assert len(filteredInputs) == len(filteredOutputs), "entry missing or index mismatch"

    return filteredInputs, filteredOutputs

"""
    Read the component configs from a TestSpec input file

    fname - (str): the name of the input file from which to read BEMP configs
"""
def getComponentsFromSpec(fname):

    # grab what we need from the input file
    FILE_OFFSET = 3
    rows = []
    with open(fname) as f:
        rd = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        for r in rd:
            rows.append(r)
    rows = rows[FILE_OFFSET:]

    p = rows.pop(0)
    p = list(p[0].split(","))
    bats = [x[0].replace(",", "") for x in rows]
    #print(len(bats))
    props = p[1:5]
    mots = p[5:9]
    cnt = p[9:]

    newRows = [] # single element is a BEMP config for 100 runs
    for b in bats:
        cmps = [b] + props + mots + cnt
        newRows.append(cmps)
    # unduplicated BEMP data to be added
    return np.asarray(newRows)


"""
    reads components from the UAV test spec csv file and add to each of the run results csv files

    cdir - (str): the location of all the csv files containing output data
    ofile - (str): the name of the file to write the monolithic dataframe
"""
def addComponentsToOutputCsv(cdir):

    RUNS = 500

    spec = 'data/UAVTestSpecAllBattMMcopy.csv'
    cmpData = getComponentsFromSpec(spec)
    assert cmpData is not None, "error reading components from spec"

    dcols = ["GUID",
    "AnalysisError",
    "Interferences",
    "Ixx",
    "iyy",
    "Izz",
    "MassEstimate",
    "Battery_amps_to_max_amps_ratio_at_MFD",
    "Battery_amps_to_max_amps_ratio_at_MxSpd",
    "Distance_at_MxSpd__m_",
    "Max_Flight_Distance__m_",
    "Max_Hover_Time__s",
    "Max_Lateral_Speed__m_s_",
    "Max_uc_at_MFD",
    "Motor_amps_to_max_amps_ratio_at_MFD",
    "Motor_amps_to_max_amps_ratio_at_MxSpd",
    "Motor_power_to_max_power_ratio_at_MFD",
    "Motor_power_to_max_power_ratio_at_MxSpd",
    "Power_at_MFD__W_",
    "Power_at_MxSpd__W_",
    "Speed_at_MFD__m_s_",
    "ArmLength",
    "BatteryOffset_X",
    "BatteryOffset_z",
    "SupportLength",
    "Battery_0",
    "Prop_0",
    "Prop_2",
    "Prop_1",
    "Prop_3",
    "Motor_0",
    "Motor_2",
    "Motor_1",
    "Motor_3",
    "ESC_0",
    "ESC_1",
    "ESC_2",
    "ESC_3"]

    flist = sorted(os.listdir(cdir))
    flist.pop(0) # remove .DS_Store
    print(f"flist is {len(flist)}")
    print(flist)
    print(f"cmpData is {len(cmpData)}")
    #assert len(cmpData) == len(flist), "length mismatch"

    # init empty dataframes
    successes = pd.DataFrame(columns=dcols)
    failures = pd.DataFrame(columns=dcols)

    # update the ESC to be ESC_Debugging- geometric properties but not electrical
    for i, c in enumerate(cmpData):
        cmpData[i][-4:] = ['ESC_Debugging', 'ESC_Debugging', 'ESC_Debugging', 'ESC_Debugging']
        #print(cmpData[i])

    for i, f in enumerate(flist): # output.csv file for each battery
        print(f"file {i+1}")
        df = pd.read_csv(os.path.join(cdir, f))

        for j, c in enumerate(dcols[-13:]):

            if i >= len(cmpData): # set i back to 0
                i = i - len(cmpData)

            df[c] = [cmpData[i][j] for _ in range(RUNS)]

        sc = df.loc[df["AnalysisError"] == False]
        fl = df.loc[df["AnalysisError"] == True]

        print(sc.columns)

        successes = successes.append(sc)
        failures = failures.append(fl)

    # successes.to_csv("data/FlightDynamicsAllBatteriesSuccessesTest.csv", index=False)
    # failures.to_csv("data/FlightDynamicsAllBatteriesFailuresTest.csv", index=False)
    print("done")


""" unzip all files from the specified directory and write files to the specified output directory """
def unzip(indir, outdir):
    os.chdir(indir)
    flist = sorted(os.listdir(indir)) # oldest to newest / battery 1-30
    flist.pop(0) # remove DS_Store

    mems = ['outputMod.csv'] # files we care about from zip archive

    for i, f in enumerate(flist):
        print(i+1)
        if f.endswith(".zip"):
            fpath = os.path.join(indir, f)
            with zipfile.ZipFile(fpath) as zf:
                for mem in mems:
                    if mem in zf.namelist():
                        if i+1 < 10:
                            num = "0" + str(i+1)
                        else:
                            num = str(i+1)
                        fname = mem[:-4] + "_b" + num + ".csv"
                        print(f" fname is {fname}")
                        fpath = os.path.join(outdir, fname)
                        with open(fpath, "wb") as fi:
                            fi.write(zf.read(mem))
    print("done")


"""
    Filter hovercalc reesults by a specified filter

    fi: filtered inputs ie no battery fried runs
    fo: filtered Outputs
    k: how many results to return for the specified criterion
"""
def sortHoverCalcResults(fi, fo, k=10, criterion='thrust'):

    print(f"crit: {criterion}")
    opts = ["thrust", "weight", "efficiency", "hover"]
    assert criterion in opts, "invalid criteria"
    assert isinstance(k, int) and k > 0, "qty > 0 required"

    """ TotalWeight,PropFolder,BatVolts,BatName,BatCapacity,BatContDisRate,NumMotor,Name,KV,KT,Rw,MaxCur,IdleCurr,ESCMaxCurr,PropName,PropDiameter,PropPerfFile """

    """
        Row,TotalMaxThrust_N,MaxDisRate_C,HoverTime_min,MaxThrottleTime_min,MotorOppTime_min,BatFried
        Row,RotSpeed_RPM,Current_A,Voltage_V,ElectricalPower_W,Thrust_N,Efficiency_%,Thrust2Weight,ESCFried
    """

    filtered = []

    # max thrust from hovercalc results
    if criterion == "thrust":
        for i, f in enumerate(fo):
            idx = f.index('Vertical') + 1 # index of thrust value
            tr = float(f[idx])
            filtered.append((i, tr))

    # max thrust2weight ratio from hovercalc results
    elif criterion == "weight":
        for i, f in enumerate(fo):
            idx = -2 # Thrust2Weight is the second to last element of input
            wt = float(f[idx])
            filtered.append((i, wt))

    # max efficiency from hovercalc results
    elif criterion == "efficiency":
        for i, f in enumerate(fo):
            idx = -3 # efficiency percentage
            ef = float(f[idx])
            filtered.append((i, ef))

    # max hover time from hovercalc results
    elif criterion == "hover":
        for i, f in enumerate(fo):
            idx = f.index('Vertical') + 3 # index of max hover time
            hv = float(f[idx])
            filtered.append((i, hv))

    # descending order i.e. filter max values
    filtered = sorted(filtered, key=itemgetter(1), reverse=True)

    # only keep the requested k
    print("idxs")
    idxs = [t[0] for t in filtered[:k]]
    print(idxs)
    print(f"---{criterion}--- values")
    vals = [t[1] for t in filtered[:k]]
    print(vals)

    # return both to enable successive filter calls
    sortedInputs = [fi[i] for i in idxs]
    sortedOutputs = [fo[i] for i in idxs]
    assert len(sortedInputs) == len(sortedOutputs), "length mismatch"

    # current component name locations in input data
    # battery [1]
    # controller [5]
    # motor [7]
    # propeller [-3]
    bmps = []
    for fin in sortedInputs:
        bidx = nameToIdx('battery', fin[1][:-4])

        c = fin[5]
        if "t-motor" in c:
            c = c[c.index("t-motor") + len("t-motor") + 1:-4]
        elif "kde_direct" in c:
            c = c[c.index("kde_direct") + len("kde_direct") + 1:-4]
        cidx = nameToIdx('controller', c)

        m = fin[7]
        if "t-motor" in m:
            m = m[m.index("t-motor") + len("t-motor") + 1:-4]
        elif "kde_direct" in m:
            m = m[m.index("kde_direct") + len("kde_direct") + 1:-4]

        midx = nameToIdx('motor', m)

        p = fin[-3]
        p = p[p.index("propellers_") + len("propellers") + 1:-4].strip()
        pidx = nameToIdx('propeller', p)

        bmp = Bemp(bidx, cidx, midx, pidx)
        bmps.append(bmp)

    return sortedInputs, sortedOutputs, bmps, vals


"""
    Read the specified file and return an ndarray of the required data

    pname: (str) the propeller data file to read from
    maxrpm: (float) the maximum rpm possible with the selected motor and battery
    propDiameter: (float) the diameter of the propeller of interest in inches
    interp: (int) the number of points to interpolate over

    returns: interpolated data for the specified propeller
"""
def parsePropFile(pname, maxrpm, propDiameter, interp=1000):

    # todo make relative path
    pname = os.path.join("/Users/michael/darpa/hovercalc-copy/HoverCalcExecution/PropData/", pname)

    TABLE_LEN = 29 # number of rows in table for each RPM
    OFFSET = 4 # number of rows between RPM and start of table data

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
            tbl = [] # table at RPM value
            for l in plines[i + OFFSET: i+OFFSET + TABLE_LEN+1]:
                tb = l[0].strip()
                tb = tb.replace(sp4, sp6)
                tb = tb.replace(sp5, sp6)
                tb = tb.replace(sp7, sp6)
                tb = tb.replace(sp8, sp6)
                tb = tb.replace("-NaN", "0") # nan hacking
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
            maxrpm = np.array([maxrpm for _ in range(NUM_MOTORS)])

    diff = rpms - np.max(maxrpm)
    idx = np.argmin(np.abs(diff))
    if idx + 1 < len(rpms):
        rpms = rpms[:idx+2] # to idx+1
    else:
        rpms = rpms[:idx+1] # to idx

    """
        dictionary containing relevant prop data based on maxrpm (dependent on motor and battery)

        keys (int): rpm values for the propeller of interest
        values (np.ndarray): a list of numpy arrays indexed from 0-7 corresponding to
                cols V, J, Pe, Ct, Cp, Pwr, Torque, Thrust, respectively

    """
    d = {rpm : tvals[i] for i, rpm in enumerate(rpms)}

    # columns- V, J, Pe, Ct, Cp, Pwr, Torque, Thrust
    thrust, cthrust, torque, power = [], [], [] , []
    for r in rpms:
        # thrust from raw data
        thr = d[r][7][0] * 4.44822 # N
        thrust.append(thr)

        # thrust from coefficient
        cth = d[r][3][0] * 1.225 * ((r/60)**2) * ((propDiameter*.0254) ** 4)
        cthrust.append(cth)

        # torque from raw data
        trq = d[r][6][0] * 0.11298482933333 # Nm
        torque.append(trq)

        # power from coefficient
        pwr = d[r][4][0] * 1.225 * ((r/60)**3) * ((propDiameter*.0254) ** 5) # Watts
        power.append(pwr)

    thrust, cthrust, torque, power = np.asarray(thrust), np.asarray(cthrust), np.asarray(torque), np.asarray(power)

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


def listComponentNames(cmpdir):
    files = set()
    for f in os.listdir(cmpdir):
        if f:
            print(f.strip())
            files.add(f.strip())
    print(len(files))
    return list(files)


""" Unpickle bemp configurations """
def unpickleConfigs(fname):
    s = time.time()
    #fname = 'bempConfigs'
    with open(fname, 'rb') as f:
        configs = pickle.load(f)
    configs = list(configs)
    #assert len(configs) == DESIGN_COUNT, "length mismatch"
    e = time.time()
    print(f"{e-s}s to load {len(configs)} configs")
    return configs


""" Store BEMP objects to a pickle file """
def pickleConfigData(bemps, i):
    #assert len(bemps) == DESIGN_COUNT, "length mismatch"
    fname = 'generatedBempData/bempConfigData_' + str(i)
    with open(fname, 'wb') as f:
        pickle.dump(bemps, f)


def unpickleData(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data

def pickleData(fname, data):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


""" return dictionary of Bemp indices to hovercalc results"""
def readData(fname):
    d = unpickleData(fname)
    return d


""" return a dataframe from the unpickled data """
def buildDataFrame(vals):
    print("vuilding df")
    # get the values
    data = []
    for k, v in tqdm(vals.items()):
        #print(k)
        thr = v[0].flatten()
        sys = v[1][:2]
        syt = v[1][2].flatten()
        k = k.split(",")
        k = [idxToAcm('battery', k[0], reverse=True),
             idxToAcm('controller', k[1], reverse=True),
             idxToAcm('motor', k[2], reverse=True),
             idxToAcm('propeller', k[3], reverse=True),
        ]
        #print(thr, sys, syt, k)
        all = np.concatenate((thr, sys, syt, k))
        data.append(all)
    assert len(data) == len(vals), "length mismatch"
    return pd.DataFrame.from_records(np.array(data, dtype=np.float32))


""" returns a dataframe of k design points filtered only max for now"""
def filterDataFrame(df, k, criterion='thrust', obj='max'):

    # power at max throttle -- idx 10
    # efficiency at max efficiency -- idx 17
    # thrust2weight at hover -- idx 18
    # total max thrust -- idx 21
    # hover time -- idx 23

    criteria = ['hover', 'thrust', 'efficiency', 'thrust2weight', 'power']

    assert criterion in criteria, "invalid criterion"
    critMap = {
        'power': 10,
        'efficiency': 17,
        'thrust2weight': 18,
        'thrust': 21,
        'hover': 23
    }

    #print(df[critMap[criterion]].head())

    # todo add support for min (change to ascending=True)
    df = df.sort_values(df.columns[critMap[criterion]], ascending=False)

    print(df[critMap[criterion]])

    return df[:k], df[critMap[criterion]][:k]

""" remove the propellers that are no longer in the corpus """
def filterConfigs(cfgs):
    skip = [264, 278, 359] # missing propellers
    return [c for c in cfgs if c[-1] not in skip]


""" unpickle the described file and write to csv with the same name """
def writeHoverCalcToCsv(fname):

    data = unpickleData(fname)

    rows = []
    keys = list(data.keys())[:5]
    vals = list(data.values())[:5]

    for k, v in data.items():
        t = np.ravel(v[0])
        # print(f"t {t}")
        s = np.ravel(v[1])
        #print(f"s {s}")
        u = np.ravel(s[-1])
        # print(f"u {u}")
        # print("concat")
        vals = list(np.concatenate((t, s[:-1], u)))
        #print(len(vals))
        bemp = k.split(",")
        #print(len(bemp+vals))
        rows.append(bemp+vals)

    header = ["Battery", "ESC", "Motor", "Propeller", "rpmHov", "rpmMax", "rpmOpp", "IinHov", "IinMax", "IinOpp", "Vhov", "Vmax", "Vopp", "PeHov", "PeMax", "PeOpp",    "thrustInHov", "thrustInMax", "thrustInOpp", "effHov", "effMax", "effOpp", "thrust2weightHov", "thrust2weightMax", "thrust2weightOpp", "totalMaxThrust", "maxC", "timeHov", "timeMax", "timeOpp"]

    fname = fname.replace("pickle", "csv")
    print(f"writing {fname}")
    with open(fname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {fname}")
    # columns of csv


def saveHoverCalcResults(readName, writeName):
        d = readData(readName)
        print("old")
        print(type(d))
        print(len(d))

        filt = filterConfigs(list(d.keys()))

        vals = [d[f] for f in filt]

        d = dict(zip(filt, vals))

        print("new")
        print(type(d))
        print(len(d))

        oldKeys = list(d.keys())
        oldValues = list(d.values())
        keys = []
        vals = []

        for k,v in tqdm(d.items()):
            #print(k)
            key = bempIdxToAcms((k[0], 0, k[1], k[2]))
            keys.append(key)
            vals.append(v)

        keys = [",".join(k) for k in keys]

        data = dict(zip(keys, vals))

        print(list(data.items())[:5])

        print(f"writing {writeName}")
        pickleData(writeName, data)
        print(f"wrote {writeName}")

def getMinArmLengthQuad(idxs):
    b, c, m, p = idxs
    bmp = Bemp(b,c,m,p)
    print(bmp)
    return bmp.minArmLength



""" Driver """

if __name__ == "__main__":

    configs = unpickleData("/Users/michael/darpa/UAVSimulyzer/generatedData/hplane/friedHplaneNoESC.pickle")
    configs = filterConfigs(configs)
    print(len(configs))

    fried = {}
    for c in tqdm(configs):
        bmp = Bemp(c[0], 0, c[1], c[2], rotors=5)
        thruster, system, batteryFried = bmp.doHoverCalc()
        if batteryFried:
            fried[c] = (thruster, system)

    fname = "/Users/michael/darpa/UAVSimulyzer/generatedData/hplane/friedHplaneNoESCresults.pickle"
    with open(fname, 'wb') as f:
        pickle.dump(fried, f)

    writeName = "/Users/michael/darpa/UAVSimulyzer/generatedData/hplane/friedHplaneNoESCdata.pickle"
    saveHoverCalcResults(fname, writeName)
    writeHoverCalcToCsv(writeName)

    # unfried = {}
    # fried = {}
    #
    # start = time.time()
    # for i, config in tqdm(enumerate(noEscConfigs)):
    #     # hardcode debugging controller idx 0 , name kde_direct_KDEXF_UAS20LV
    #     #print(config)
    #     b = Bemp(config[0], 0, config[1], config[2], rotors=5)
    #     #print(b)
    #     s, t = b.doHoverCalc()
    #
    #     if isinstance(s, np.ndarray) and isinstance(t, np.ndarray):
    #         unfried[config] = (s, t)
    #     else:
    #         fried.append(config)
    #
    # uname = 'unfriedHplaneNoESC.pickle'
    # with open(uname, 'wb') as u:
    #     pickle.dump(unfried, u)
    #

    #
    # print(f"len unfried {len(unfried.keys())}")
    # print(f"len fried {len(fried)}")
    #
    # end = time.time()
    # print(f" done in {end-start}")

    #readName = "unfriedHplaneNoESC.pickle"
    #writeName = "/Users/michael/darpa/UAVSimulyzer/unfriedHplaneNoESC500k.pickle"
    #saveHoverCalcResults(readName, writeName)

    #data = unpickleData("/Users/michael/darpa/UAVSimulyzer/unfriedQuad_2M_MinLength.pickle")

    #fname = "/Users/michael/darpa/UAVSimulyzer/unfriedHQuadPlaneNoESC.pickle"
    #writeHoverCalcToCsv(writeName)
