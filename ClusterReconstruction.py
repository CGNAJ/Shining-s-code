import numpy as np
import pandas as pd
import random
import math
import csv
from scipy.optimize import root, fsolve
import matplotlib.pyplot as plt
import math
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
event = pd.read_csv('./NewGenerate.csv', header = None)

def DataPrepare():										#Read in all data from dataset
	output = np.zeros((15000, 6, 12))					#Colume 1: accurate x coordinates
	for i in range(15000):								#Colume 2: number of muon hit strip 
		for j in range(6):								#Colume 3: momentum of the muon
			output[i][j] = event.values[j+6*i]			#Colume 4-11: number of white noise strip
			#print(output[i][j])						#Colume 12: number of side-hit strip
	return output										#Return a [number of events][6][12] array

def RPC2_strips(event_data):
	strips = np.zeros((15000, 20))
	for i in range(15000):								#Delete 'accurate coordinates' and 'momentum' from dataset
		strips[i][0] = event_data[i][2][1]				#All position of strips with signals are reserved
		strips[i][1] = event_data[i][3][1]				#Record RPC2-1 and RPC2-2 signals in one array
		for k in range(8):
			strips[i][2+k] = event_data[i][2][3+k]
			strips[i][10+k] = event_data[i][3][3+k]
		strips[i][18] = event_data[i][2][11]
		strips[i][19] = event_data[i][3][11]
	return strips 										#Return Array

def RPC1_strips(event_data):
	strips = np.zeros((15000, 20))
	for i in range(15000):								#Delete 'accurate coordinates' and 'momentum' from dataset
		strips[i][0] = event_data[i][0][1]				#All position of strips with signals are reserved
		strips[i][1] = event_data[i][1][1]				#Record RPC1-1 and RPC1-2 signals in one array
		for k in range(8):
			strips[i][2+k] = event_data[i][0][3+k]
			strips[i][10+k] = event_data[i][1][3+k]
		strips[i][18] = event_data[i][0][11]
		strips[i][19] = event_data[i][1][11]
	return strips  										#Return Array

def RPC3_strips(event_data):
	strips = np.zeros((15000, 20))
	for i in range(15000):								#Delete 'accurate coordinates' and 'momentum' from dataset
		strips[i][0] = event_data[i][4][1]				#All position of strips with signals are reserved
		strips[i][1] = event_data[i][5][1]				#Record RPC3-1 and RPC3-2 signals in one array
		for k in range(8):
			strips[i][2+k] = event_data[i][4][3+k]
			strips[i][10+k] = event_data[i][5][3+k]
		strips[i][18] = event_data[i][4][11]
		strips[i][19] = event_data[i][5][11]
	return strips

def cluster_Search(RPC):								#Original Version  List out all clusters' position
	Cluster_Pos_Array = []
	for i in range (15000):									#Scan all events
		Cluster_Pos = []
		RPC[i] = np.sort(RPC[i])							#All the signals are sorted from small strip number to large strip number
		for j in RPC[i]:									#!!!Delete all single signals
			if j!=0:													#Pick out strips with signals
				cluster_signal_right = list(np.where(RPC[i] == j+1))	#Check if there is another signal to the right side of the original signal within one strip
				cluster_signal_equal = list(np.where(RPC[i] == j))		#Check if there is another signal on the top or bottom of the original signal
				cluster_signal_left = list(np.where(RPC[i] == j-1))		#Check if there is another signal to the left side of the original signal within one strip
				if len(cluster_signal_right[0]) == 0 and (len(cluster_signal_equal[0])-1) == 0 and len(cluster_signal_left[0]) == 0:
					RPC[i][np.where(RPC[i] == j)] = 0					#Delete single signals
					RPC[i] = np.sort(RPC[i])							#Sort all signals
		for j in RPC[i]:									#This part is considering conditions when the length of cluster is longer than 2 strips							
			x = 0
			if j!=0:
				cluster_signal_right2 = list(np.where(RPC[i] == j+2))	#Check if there is another signal to the right side of the original signal within two strips
				cluster_signal_right = list(np.where(RPC[i] == j+1))
				cluster_signal_equal = list(np.where(RPC[i] == j))		#Check if there is another signal on the top or bottom of the original signal
				cluster_signal_left = list(np.where(RPC[i] == j-1))		#Check if there is another signal to the left side of the original signal within two strips
				cluster_signal_left2 = list(np.where(RPC[i] == j-2))
				x = 0.03*(len(cluster_signal_right2[0])*(j+2)+len(cluster_signal_right[0])*(j+1)+len(cluster_signal_equal[0])*j+len(cluster_signal_left[0])*(j-1)+len(cluster_signal_left2[0])*(j-2))/(len(cluster_signal_right[0])+len(cluster_signal_equal[0])+len(cluster_signal_left[0])+len(cluster_signal_left2[0])+len(cluster_signal_right2[0]))
					#print(x)
			if x!=0 and x not in Cluster_Pos:
				Cluster_Pos.append(x)
		Cluster_Pos_Array.append(Cluster_Pos)
	#print(Cluster_Pos_Array)
	return Cluster_Pos_Array

def cluster_reconstruction(Data, RPC1, RPC2, RPC3, RPC1_Cluster_Pos, RPC2_Cluster_Pos, RPC3_Cluster_Pos):
	reconstructed_cluster_pos = []
	diff_distribution = []
	Difference_record1 = 0
	Difference_record3 = 0
	momentum = []
	count_reconstructed = [0, 0, 0]
	for i in range(15000):								#Scan all events
		cluster_reconstruction_pos = [0, 0, 0]
		if sorted(RPC1[i])[19] == 0 or sorted(RPC2[i])[19] == 0 or sorted(RPC3[i])[19] == 0:	#if one layer of the event contains no signal, delete the event
			continue
		else:
			pass
		seed_x = 0
		scan_1 = 8											#scanning window
		scan_3 = 16											#scanning window
		min_RPC1_x = 0
		min_RPC2_x = 0
		min_RPC3_x = 0			
		Total_diff = 1000									#Difference flag
		if(len(RPC2_Cluster_Pos[i])!=0):					# Condition1: There is 1 or more cluster on RPC2
			for j in range(len(RPC2_Cluster_Pos[i])):			#For each cluster, calculate the intersection coordinates between RPC1&RPC3 and the line
				min_RPC1_x_serial = 0							#between original point and cluster signal on RPC2
				min_RPC3_x_serial = 0
				seed_x = RPC2_Cluster_Pos[i][j]
				seed_y = 7.481
				#intersection points
				intersection_RPC1_x = 6.803*seed_x/seed_y
				intersection_RPC3_x = 9.835*seed_x/seed_y		#Intersection points coordinates
				if(len(RPC1_Cluster_Pos[i])!=0):					#If there is 1 or more cluster on RPC1
					diff1 = 1000										#Difference flag 1
					for k in RPC1_Cluster_Pos[i]:
						if k == 0 or math.fabs(k - intersection_RPC1_x) > 0.48:	#Pass signals outside the scanning window
							continue
						if(math.fabs(k - intersection_RPC1_x) < diff1):	#If difference of new cluster is smaller than difference flag
							diff1 = math.fabs(k - intersection_RPC1_x)	#then difference flag = the new smaller difference
							min_RPC1_x_serial = k						#Record the smaller difference cluster
					#cluster_reconstruction_pos[0] = RPC1_Cluster_Pos[i][min_RPC1_x_serial]
				else:												#If there isn't any cluster on RPC1
					diff1 = 1000										#Difference flag 1
					for k in range(20):									#Scan all single signals
						if RPC1[i][k] == 0 or math.fabs(0.03*RPC1[i][k] - intersection_RPC1_x) > 0.48:	#Pass signals outside the scanning window
							continue
						if(math.fabs(0.03*RPC1[i][k] - intersection_RPC1_x) < diff1):	#If difference of new cluster is smaller than difference flag
							diff1 = math.fabs(0.03*RPC1[i][k] - intersection_RPC1_x)	#difference flag = the new smaller difference
							min_RPC1_x_serial = RPC1[i][k]								#Record the smaller difference cluster
					#cluster_reconstruction_pos[0] = 0.03*RPC1[i][min_RPC1_x_serial]
				if(len(RPC3_Cluster_Pos[i])!=0):
					diff3 = 1000										
					for k in RPC3_Cluster_Pos[i]:
						if k == 0 or (k - intersection_RPC3_x) < -0.72 or (k - intersection_RPC3_x) > 0.96:
							continue
						if(math.fabs(k - intersection_RPC3_x) < diff3):
							diff3 = math.fabs(k - intersection_RPC3_x)
							min_RPC3_x_serial = k
					#cluster_reconstruction_pos[2] = RPC3_Cluster_Pos[i][min_RPC3_x_serial]
				else:
					diff3 = 1000
					for k in range(20):
						if RPC3[i][k] == 0 or (0.03*RPC3[i][k] - intersection_RPC3_x) < -0.72 or (0.03*RPC3[i][k] - intersection_RPC3_x) > 0.96:
							continue
						if(math.fabs(0.03*RPC3[i][k] - intersection_RPC3_x) < diff3):
							diff3 = math.fabs(0.03*RPC3[i][k] - intersection_RPC3_x)
							min_RPC3_x_serial = RPC3[i][k]
					#cluster_reconstruction_pos[2] = 0.03*RPC3[i][min_RPC3_x_serial]
				if(diff1 + diff3 < Total_diff):					#If diff1 + diff3 is smaller than difference flag
					Total_diff = diff1 + diff3 					#difference flag = the new smaller difference
					min_RPC2_x = RPC2_Cluster_Pos[i][j]			#Record the smaller difference clusters
					cluster_reconstruction_pos[1] = min_RPC2_x
					if len(RPC1_Cluster_Pos[i])!=0 and min_RPC1_x_serial != 0:	
						cluster_reconstruction_pos[0] = min_RPC1_x_serial
					elif len(RPC1_Cluster_Pos[i]) == 0 and min_RPC1_x_serial != 0:
						cluster_reconstruction_pos[0] = 0.03*min_RPC1_x_serial
					else:
						continue
					if len(RPC3_Cluster_Pos[i])!=0 and min_RPC3_x_serial != 0:
						cluster_reconstruction_pos[2] = min_RPC3_x_serial
					elif len(RPC3_Cluster_Pos[i]) == 0 and min_RPC3_x_serial != 0:
						cluster_reconstruction_pos[2] = 0.03*min_RPC3_x_serial
					else:
						continue
			if cluster_reconstruction_pos[0] == 0 or cluster_reconstruction_pos[1] == 0 or cluster_reconstruction_pos[2] == 0:
				continue
			momentum.append(Data[i][0][2])						#Count reconstructed clusters
			if(len(RPC1_Cluster_Pos[i])==0):
				count_reconstructed[0] = count_reconstructed[0] + 1
			if(len(RPC2_Cluster_Pos[i])==0):
				count_reconstructed[1] = count_reconstructed[1] + 1
			if(len(RPC3_Cluster_Pos[i])==0):
				count_reconstructed[2] = count_reconstructed[2] + 1
			diff_distribution.append((cluster_reconstruction_pos[0] - intersection_RPC1_x)/0.03+(cluster_reconstruction_pos[2] - intersection_RPC3_x)/0.03)
			reconstructed_cluster_pos.append(cluster_reconstruction_pos)
		else:
			for j in range(20):
				min_RPC1_x_serial = 0
				min_RPC3_x_serial = 0
				if(RPC2[i][j]!=0):
					seed_x = RPC2[i][j]
					seed_y = 7.481
					#intersection points
					intersection_RPC1_x = 6.803*seed_x/seed_y
					intersection_RPC3_x = 9.835*seed_x/seed_y
					if(len(RPC1_Cluster_Pos[i])!=0):
						diff1 = 1000
						for k in RPC1_Cluster_Pos[i]:
							if k == 0 or math.fabs(k - intersection_RPC1_x) > 0.48:
								continue
							if(math.fabs(k - intersection_RPC1_x) < diff1):
								diff1 = math.fabs(k - intersection_RPC1_x)
								min_RPC1_x_serial = k
						#cluster_reconstruction_pos[0] = RPC1_Cluster_Pos[i][min_RPC1_x_serial]
					else:
						diff1 = 1000
						for k in range(20):
							if RPC1[i][k] == 0 or math.fabs(0.03*RPC1[i][k] - intersection_RPC1_x) > 0.48:
								continue
							if(math.fabs(0.03*RPC1[i][k] - intersection_RPC1_x) < diff1):
								diff1 = math.fabs(0.03*RPC1[i][k] - intersection_RPC1_x)
								min_RPC1_x_serial = RPC1[i][k]
						#cluster_reconstruction_pos[0] = 0.03*RPC1[i][min_RPC1_x_serial]

					if(len(RPC3_Cluster_Pos[i])!=0):
						diff3 = 1000
						for k in RPC3_Cluster_Pos[i]:
							if k == 0 or (k - intersection_RPC3_x) < -0.72 or (k - intersection_RPC3_x) > 0.96:
								continue
							if(math.fabs(k - intersection_RPC3_x) < diff3):
								diff3 = math.fabs(k - intersection_RPC3_x)
								min_RPC3_x_serial = k
						#cluster_reconstruction_pos[2] = RPC3_Cluster_Pos[i][min_RPC3_x_serial]
					else:
						diff3 = 1000
						for k in range(20):
							if RPC3[i][k] == 0 or (0.03*RPC3[i][k] - intersection_RPC3_x) < -0.72 or (0.03*RPC3[i][k] - intersection_RPC3_x) > 0.96:
								continue
							if(math.fabs(0.03*RPC3[i][k] - intersection_RPC3_x) < diff3):
								diff3 = math.fabs(0.03*RPC3[i][k] - intersection_RPC3_x)
								min_RPC3_x_serial = RPC3[i][k]
						#cluster_reconstruction_pos[2] = 0.03*RPC3[i][min_RPC3_x_serial]
					if(diff1 + diff3 < Total_diff):
						Total_diff = diff1 + diff3
						#print(Total_diff)
						min_RPC2_x = 0.03*RPC2[i][j]
						cluster_reconstruction_pos[1] = min_RPC2_x
					#Difference_record1 = diff1
					#Difference_record3 = diff3
						if len(RPC1_Cluster_Pos[i])!=0 and min_RPC1_x_serial != 0:
							cluster_reconstruction_pos[0] = min_RPC1_x_serial
						elif len(RPC1_Cluster_Pos[i]) == 0 and min_RPC1_x_serial != 0:
							cluster_reconstruction_pos[0] = 0.03*min_RPC1_x_serial
						else:
							continue
						if len(RPC3_Cluster_Pos[i])!=0 and min_RPC3_x_serial != 0:
							cluster_reconstruction_pos[2] = min_RPC3_x_serial
						elif len(RPC3_Cluster_Pos[i]) == 0 and min_RPC3_x_serial != 0:
							cluster_reconstruction_pos[2] = 0.03*min_RPC3_x_serial
						else:
							continue
			if cluster_reconstruction_pos[0] == 0 or cluster_reconstruction_pos[1] == 0 or cluster_reconstruction_pos[2] == 0:
				continue
			#print(cluster_reconstruction_pos)
			momentum.append(Data[i][0][2])
			if(len(RPC1_Cluster_Pos[i])==0):
				count_reconstructed[0] = count_reconstructed[0] + 1
			if(len(RPC2_Cluster_Pos[i])==0):
				count_reconstructed[1] = count_reconstructed[1] + 1
			if(len(RPC3_Cluster_Pos[i])==0):
				count_reconstructed[2] = count_reconstructed[2] + 1
			diff_distribution.append((cluster_reconstruction_pos[0] - intersection_RPC1_x)/0.03+(cluster_reconstruction_pos[2] - intersection_RPC3_x)/0.03)
			reconstructed_cluster_pos.append(cluster_reconstruction_pos)
	#print(np.array(reconstructed_cluster_pos))
	#print(np.array(diff_distribution))
	return(np.array(reconstructed_cluster_pos), np.array(diff_distribution), np.array(momentum), np.array(count_reconstructed))

if __name__ == '__main__':
	Data = DataPrepare()
	RPC1 = RPC1_strips(Data)
	RPC2 = RPC2_strips(Data)
	RPC3 = RPC3_strips(Data)
	RPC1_Cluster_Pos = cluster_Search(RPC1)
	RPC2_Cluster_Pos = cluster_Search(RPC2)
	RPC3_Cluster_Pos = cluster_Search(RPC3)
	RPC11 = RPC1_strips(Data)
	RPC22 = RPC2_strips(Data)
	RPC33 = RPC3_strips(Data)
	Reconstruction = cluster_reconstruction(Data, RPC11, RPC22, RPC33, RPC1_Cluster_Pos, RPC2_Cluster_Pos, RPC3_Cluster_Pos)
	#print(Data)
	hits_count = [0, 0, 0]
	Cluster_count = np.zeros((3, 4))
	Total_hits_count = [0, 0, 0, 0, 0, 0, 0]
	Total_clusters_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	Total_hits_and_clusters_count = [0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	#print(RPC1_Cluster_Pos)
	hit_count = 0
	cluster_count = 0
	hit_and_cluster_count = 0
	print(Reconstruction[3])
	for i in range(15000):
		if len(RPC1_Cluster_Pos[i]) == 0:
			Cluster_count[0][0] = Cluster_count[0][0] + 1
		elif len(RPC1_Cluster_Pos[i]) == 1:
			Cluster_count[0][1] = Cluster_count[0][1] + 1
		elif len(RPC1_Cluster_Pos[i]) == 2:
			Cluster_count[0][2] = Cluster_count[0][2] + 1
		else:
			pass
		if len(RPC2_Cluster_Pos[i]) == 0:
			Cluster_count[1][0] = Cluster_count[1][0] + 1
		elif len(RPC2_Cluster_Pos[i]) == 1:
			Cluster_count[1][1] = Cluster_count[1][1] + 1
		elif len(RPC2_Cluster_Pos[i]) == 2:
			Cluster_count[1][2] = Cluster_count[1][2] + 1
		else:
			pass
		if len(RPC3_Cluster_Pos[i]) == 0:
			Cluster_count[2][0] = Cluster_count[2][0] + 1
		elif len(RPC3_Cluster_Pos[i]) == 1:
			Cluster_count[2][1] = Cluster_count[2][1] + 1
		elif len(RPC3_Cluster_Pos[i]) == 2:
			Cluster_count[2][2] = Cluster_count[2][2] + 1
		else:
			pass
		'''if(Data[i][0][0]!=0):
			#hit_count = hit_count + 1
			hit_and_cluster_count = hit_and_cluster_count + 1
			hits_count[0] = hits_count[0] + 1
		if(Data[i][1][0]!=0):
			#hit_count = hit_count + 1
			hit_and_cluster_count = hit_and_cluster_count + 1
			hits_count[0] = hits_count[0] + 1
		if(Data[i][2][0]!=0):
			#hit_count = hit_count + 1
			hit_and_cluster_count = hit_and_cluster_count + 1
			hits_count[1] = hits_count[1] + 1
		if(Data[i][3][0]!=0):
			#hit_count = hit_count + 1
			hit_and_cluster_count = hit_and_cluster_count + 1
			hits_count[1] = hits_count[1] + 1
		if(Data[i][4][0]!=0):
			#hit_count = hit_count + 1
			hit_and_cluster_count = hit_and_cluster_count + 1
			hits_count[2] = hits_count[2] + 1
		if(Data[i][5][0]!=0):
			#hit_count = hit_count + 1
			hit_and_cluster_count = hit_and_cluster_count + 1
			hits_count[2] = hits_count[2] + 1
		Total_hits_count[int(hit_count)] = Total_hits_count[int(hit_count)] + 1
		cluster_count = cluster_count + len(RPC1_Cluster_Pos[i]) + len(RPC2_Cluster_Pos[i]) + len(RPC3_Cluster_Pos[i])
		#print(len(RPC1_Cluster_Pos[i]))
		#print(len(RPC2_Cluster_Pos[i]))
		#print(len(RPC3_Cluster_Pos[i]))
		#print("next")
		hit_and_cluster_count = hit_and_cluster_count + len(RPC1_Cluster_Pos[i]) + len(RPC2_Cluster_Pos[i]) + len(RPC3_Cluster_Pos[i])
		if cluster_count > 9:
			continue
		Total_clusters_count[int(cluster_count)] = Total_clusters_count[int(cluster_count)] + 1
		Total_hits_and_clusters_count[int(hit_and_cluster_count)] = Total_hits_and_clusters_count[int(hit_and_cluster_count)] + 1
	#print(hits_count)
	print(Total_hits_count)
	print(Total_clusters_count)
	print(Total_hits_and_clusters_count)
	#print(Reconstruction[0])
	#print(Reconstruction[1])
	#print(Reconstruction[2])'''
	plt.figure()
	plt.scatter(Reconstruction[2], Reconstruction[1], s = 0.3)
	plt.ylim(0, 20)
	plt.xlabel("Absolute Value of Momentum of Muon-")
	plt.ylabel("Difference/m")
	plt.figure()
	plt.bar(x = [1, 2, 3], height = Reconstruction[3])
	plt.xlabel('RPC serial')
	plt.ylabel('Number of Clusters reconstructed')
	for a,b in zip([1, 2, 3], Reconstruction[3]):
		plt.text(a-0.4,b+0.1,'%.0f'%b)
	'''plt.figure()
	plt.bar(x = np.arange(7), height = Total_hits_count)
	plt.xlabel('number of hits')
	plt.ylabel('count')
	for a,b in zip(np.arange(7), Total_hits_count):
		plt.text(a-0.4,b+0.1,'%.0f'%b)
	plt.figure()
	plt.bar(x = np.arange(3), height = hits_count)
	plt.xlabel('order of RPC')
	plt.ylabel('number of hits')
	plt.ylim(0,30000)
	for a,b in zip(np.arange(3), hits_count):
		plt.text(a-0.4,b+0.1,'%.0f'%b)
	plt.figure()
	plt.bar(x = np.arange(10), height = Total_clusters_count)
	plt.xlabel('number of clusters')
	plt.ylabel('count')
	for a,b in zip(np.arange(10), Total_clusters_count):
		plt.text(a-0.4,b+0.1,'%.0f'%b)
	plt.figure()
	plt.bar(x = np.arange(17), height = Total_hits_and_clusters_count)
	plt.xlabel('number of hits and clusters')
	plt.ylabel('count')
	for a,b in zip(np.arange(17), Total_hits_and_clusters_count):
		plt.text(a-0.4,b+0.1,'%.0f'%b)
	plt.figure()
	print(Cluster_count[0])
	plt.bar(x = np.arange(4), height = Cluster_count[0])
	plt.xlabel('Clusters on RPC1')
	plt.ylabel('count')
	for a,b in zip(np.arange(4), Cluster_count[0]):
		plt.text(a-0.4,b+0.1,'%.0f'%b)
	plt.figure()
	plt.bar(x = np.arange(4), height = Cluster_count[1])
	plt.xlabel('Clusters on RPC2')
	plt.ylabel('count')
	for a,b in zip(np.arange(4), Cluster_count[1]):
		plt.text(a-0.4,b+0.1,'%.0f'%b)
	plt.figure()
	plt.bar(x = np.arange(4), height = Cluster_count[2])
	plt.xlabel('Clusters on RPC3')
	plt.ylabel('count')
	for a,b in zip(np.arange(4), Cluster_count[2]):
		plt.text(a-0.4,b+0.1,'%.0f'%b)'''
	plt.show()
