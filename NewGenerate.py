'''
	
	Output：Each event has 6 lines，Each line include：muon hit coordinates，muon hit strips，momentum，noise strips，side-hit strip

'''
import numpy as np
import random
import math
import csv
from scipy.optimize import root, fsolve
import matplotlib.pyplot as plt
import math

PATH = "./Test.csv"
event_num = 15000
noise_possibility = 0.001
cluster_possibility = 0.35
h = 6.8

def RandomIncidenceAngle(AngleMin, AngleMax):				#Random insert angle
	angle = random.uniform(AngleMin*math.pi/180, AngleMax*math.pi/180)
	return angle

def RandomIncidenceMomentum(MomentumMin, MomentumMax):		#Random insert momentum
	momentum = random.uniform(MomentumMin, MomentumMax)
	return momentum

def RPCXcoordinates(momentum, angle, i):					#Calculate muon hits position on all RPCs
	list_height = [6.806, 7.478, 7.484, 9.832, 9.838]
	Radius = 20*momentum/3
	circle_center_x = h*math.tan(angle) + Radius*math.cos(angle)
	circle_center_y = h - Radius*math.sin(angle)
	phi = math.asin((list_height[i]-circle_center_y)/Radius)
	hit_x = circle_center_x - Radius*math.cos(phi)
	return hit_x

def Trace():												#Draw the track of muons
	'''RPC_3_2[4] = {0, 9.838, 12.267, 9.838}
	RPC_3_1[4] = {0, 9.832, 12.267, 9.832}
	RPC_2_2[4] = {0, 7.484, 9.66, 7.484}
	RPC_2_1[4] = {0, 7.478, 9.66, 7.478}
	RPC_1_2[4] = {0, 6.806, 9.147, 6.806}
	RPC_1_1[4] = {0, 6.8, 9.147, 6.8}  '''

	min_angle = 10
	max_angle = 40
	min_momentum = 3
	max_momentum = 25
	momentum = RandomIncidenceMomentum(min_momentum, max_momentum)
	angle = RandomIncidenceAngle(min_angle, max_angle)

	B = 0.5
	e = 1.60217663410 * 10**-19
	Radius = 20*momentum/3

	strip_x = [0, 0, 0, 0, 0, 0]
	RPCx = [h*math.tan(angle), RPCXcoordinates(momentum, angle, 0), RPCXcoordinates(momentum, angle, 1), RPCXcoordinates(momentum, angle, 2), RPCXcoordinates(momentum, angle, 3), RPCXcoordinates(momentum, angle, 4)]
	for i in range(6):
		rand = random.uniform(0, 1)
		strip_x[i] = int(RPCx[i]/0.03)
		#print(momentum)

	return [RPCx, strip_x, momentum, angle]

def white_noise_generate(noise_possibility):				#Noise generaton
	list_RPC_width = [9.147, 9.147, 9.66, 9.66, 12.267, 12.267]
	list_num_of_strip = [305, 305, 322, 322, 408, 408]
	noise_pos_number = np.zeros((6, 8))
	for i in range (6):
		count = 0
		for j in range (list_num_of_strip[i]):
			rand = random.uniform(0, 1)
			if(rand<0.001):
				noise_pos_number[i][count] = j
				count = count + 1
				if(count==8):
					break
		#print(noise_pos_number[i])
	return noise_pos_number

'''def cluster_generate(RPCx, strip_x, cluster_possibility):		#不需考虑低能粒子信号，仅考虑普通信号引起的cluster
	cluster_xstrip = [0, 0, 0, 0, 0, 0]
	for i in range(6):
		if(RPCx[i]>0):
			check = RPCx[i] - 0.03*strip_x[i]
			if(RPCx[i]!=0):	
				if(check<=0.015):
					rand1 = random.uniform(0, 1)
					if(rand1<=cluster_possibility):
						cluster_xstrip[i] = strip_x[i] - 1
					else:
						cluster_xstrip[i] = 0
				else:
					rand2 = random.uniform(0, 1)
					if(rand2<=cluster_possibility):
						cluster_xstrip[i] = strip_x[i] + 1
					else:
						cluster_xstrip[i] = 0
			else:
				cluster_xstrip[i] = 0
	return cluster_xstrip		#输出有cluster的strip位置'''

def cluster_generate(RPCx, strip_x, cluster_possibility):		#Cluster Generation
	cluster_xstrip = 0
	if(RPCx>0):
		check = RPCx - 0.03*strip_x	
		if(check<=0.015):
			rand1 = random.uniform(0, 1)
			if(rand1<=cluster_possibility):
				cluster_xstrip = strip_x - 1
			else:
				cluster_xstrip = 0
		else:
			rand2 = random.uniform(0, 1)
			if(rand2<=cluster_possibility):
				cluster_xstrip = strip_x + 1
			else:
				cluster_xstrip = 0
	else:
		cluster_xstrip = 0
	return cluster_xstrip

def DataGenerate():
	output_list = np.zeros((event_num, 6, 12))
	for i in range (event_num):
		Tracedata = Trace()
		print(Tracedata[2])
		Noisedata = white_noise_generate(noise_possibility)
		#Clusterdata = cluster_generate(Tracedata[0], Tracedata[1], cluster_possibility)
		for j in range(6): 
			rand = random.uniform(0, 1)
			if(rand > 0.3):
				output_list[i][j][0] = Tracedata[0][j]
				output_list[i][j][1] = Tracedata[1][j]
			else:
				output_list[i][j][0] = 0
				output_list[i][j][1] = 0
			output_list[i][j][2] = Tracedata[2]
			for k in range(8):
				output_list[i][j][3+k] = Noisedata[j][k]
			output_list[i][j][11] = cluster_generate(Tracedata[0][j], Tracedata[1][j], cluster_possibility)

	return output_list
	#print(output_list)

if __name__ == '__main__':
	output = DataGenerate()
	list_RPC_height = [6.8, 6.806, 7.478, 7.484, 9.832, 9.838]
	file = open("./NewGenerate.csv","w",newline = '')
	csv_writer = csv.writer(file)
	print(output)
	for i in range(event_num):
		for j in range(6):
			csv_writer.writerow(output[i][j])

	seed = random.randint(0, 14999)
	plt.figure()
	plt.plot([0,9.147],[6.8,6.8], c = 'black', linewidth = 0.2)
	plt.plot([0,9.147],[6.806,6.806], c = 'black', linewidth = 0.2)
	plt.plot([0,9.66],[7.478,7.478], c = 'black', linewidth = 0.2)
	plt.plot([0,9.66],[7.484,7.484], c = 'black', linewidth = 0.2)
	plt.plot([0,12.267],[9.832,9.832], c = 'black', linewidth = 0.2)
	plt.plot([0,12.267],[9.832,9.832], c = 'black', linewidth = 0.2)
	plt.axis([0, 15, 0, 15])
	for i in range(6):
		plt.plot(output[seed][i][1]*0.03, list_RPC_height[i], 'bo', markersize = '1.2')
		for j in range(8):
			plt.plot(output[seed][i][3+j]*0.03, list_RPC_height[i], 'ro', markersize = '0.8')
	plt.plot(output[seed][i][11]*0.03, list_RPC_height[i], 'go', markersize = '0.8')
	plt.plot([0,output[seed][0][1]*0.03],[0,6.8], c = 'red', linewidth = 0.4)
	hit_count = [0, 0, 0, 0, 0, 0]
	noise_count = np.zeros((15000,6))
	noise_total_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	'''for i in range(15000):
		if(output[i][0][0] != 0):
			hit_count[0] = hit_count[0]+1
		if(output[i][1][0] != 0):
			hit_count[1] = hit_count[1]+1
		if(output[i][2][0] != 0):
			hit_count[2] = hit_count[2]+1
		if(output[i][3][0] != 0):
			hit_count[3] = hit_count[3]+1
		if(output[i][4][0] != 0):
			hit_count[4] = hit_count[4]+1
		if(output[i][5][0] != 0):
			hit_count[5] = hit_count[5]+1
		for j in range(10):
			if(output[i][0][3+j] != 0):
				noise_count[i][0] = noise_count[i][0] + 1
			if(output[i][1][3+j] != 0):
				noise_count[i][1] = noise_count[i][1] + 1
			if(output[i][2][3+j] != 0):
				noise_count[i][2] = noise_count[i][2] + 1
			if(output[i][3][3+j] != 0):
				noise_count[i][3] = noise_count[i][3] + 1
			if(output[i][4][3+j] != 0):
				noise_count[i][4] = noise_count[i][4] + 1
			if(output[i][5][3+j] != 0):
				noise_count[i][5] = noise_count[i][5] + 1
		m = noise_count[i][0]+noise_count[i][1]+noise_count[i][2]+noise_count[i][3]+noise_count[i][4]+noise_count[i][5]
		if(m>10):
			continue
		noise_total_count[int(m)] += 1
	plt.figure()
	plt.bar(x = np.arange(10), height = noise_total_count)
	plt.xlabel('number of noise')
	plt.ylabel('count')
	for a,b in zip(np.arange(10), noise_total_count):
		plt.text(a-0.4,b+0.1,'%.0f'%b)
	plt.figure()
	plt.bar(x = np.arange(6), height = hit_count)
	plt.xlabel('order of RPC')
	plt.ylabel('number of hits')
	plt.ylim(0,15000)
	for a,b in zip(np.arange(6), hit_count):
		plt.text(a-0.4,b+0.1,'%.0f'%b)
	print(hit_count)
	print(noise_count)
	print(noise_total_count)'''
	plt.show()