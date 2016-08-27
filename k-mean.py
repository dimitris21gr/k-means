import pandas as pd
import sys
import re
import copy
from random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def InitializeClusterCenters(max_val, termVector):
	centers = []
	rands = []
	for i in range(0, 5):
		j = randint(0, max_val - 1)
		while j in rands:
			j = randint(0, max_val - 1)
		rands.append(j)
		centers.append(list(termVector[j]))
	#print rands	#print to see initial centroids
	return centers

def ClosestClusterCenter(centers, d, termVector):
	bestScore = 0
	closestCenter = 0
	for i in range(0, 5):
		score = cosine_similarity(termVector[d], centers[i])[0][0]
		if (score > bestScore):
			bestScore = score
			closestCenter = i
	return closestCenter
	
def CalculateClusterMean(clusters, c, termVector, dim):
	total = len(clusters[c])
	newCenter = []
	for i in range(0, dim):
		summary = 0
		for j in range(0, total):
			summary += termVector[clusters[c][j]][i]
		newCenter.append(summary/total)
	return newCenter


#Read Data
sys.stdout.write("Loading csv file...")
sys.stdout.flush()
df=pd.read_csv("train_set.csv",sep="\t")
sys.stdout.write("done!\n")
sys.stdout.flush()

sys.stdout.write("Preprocessing data...")
sys.stdout.flush()
amount = df["Id"].size
documents = []
documentCategory = []

for i in range(0, amount):
	letters = re.sub("[^a-zA-Z]", " ", df["Content"][i])	#delete punctuation and numbers
	documents.append(letters)
	documentCategory.append(df["Category"][i])

vectorizer = CountVectorizer(stop_words='english')
data = vectorizer.fit_transform(documents)

transformer = TfidfTransformer()
transformedData = transformer.fit_transform(data)

svd = TruncatedSVD(n_components=50, random_state=42)
termVector = svd.fit_transform(transformedData)
sys.stdout.write("done!\n")
sys.stdout.flush()

clusters = [[], [], [], [], []]

centers = InitializeClusterCenters(amount, termVector)

iteration = 1
sys.stdout.write("Running K-means algorithm with random initial centroids...\nIteration: " + str(iteration))
sys.stdout.flush()
while (True):

	if (iteration > 1 and iteration < 10):
		sys.stdout.write("\b" + str(iteration))
	elif (iteration >= 10):
		sys.stdout.write("\b\b" + str(iteration))
	sys.stdout.flush()
	
	oldCenters = copy.deepcopy(centers)
	
	for i in range(0, amount):	#group documents to clusters
		idx = ClosestClusterCenter(centers, i, termVector)
		clusters[idx].append(i)
		
	centers = []
	for i in range(0, 5):		#find new centers by calculating mean values of each cluster
		centers.append(CalculateClusterMean(clusters, i, termVector, 50))
		
	stop = True
	for i in range(0, 5):		#check if new centers are the same with the old ones
		if (cmp(oldCenters[i], centers[i]) != 0):
			stop = False
			break
		
	if (stop == True):			#if all 5 centers remain the same, we are done
		break

	clusters = [[], [], [], [], []]		#if not, build clusters again according to the new centers
	iteration += 1

results = [[], [], [], [], []]
for i in range(0,5):
	total = len(clusters[i])
	categorySum = [0, 0, 0, 0, 0]
	for j in range(0, total):
		if documentCategory[clusters[i][j]][0] == "P":
			categorySum[0] += 1
		elif documentCategory[clusters[i][j]][0] == "B":
			categorySum[1] += 1
		elif documentCategory[clusters[i][j]][0] == "T":
			categorySum[2] += 1
		elif documentCategory[clusters[i][j]] == "Film":
			categorySum[3] += 1
		else:
			categorySum[4] += 1
	for j in range(0, 5):
		categorySum[j] = categorySum[j] / float(total)		#get percentage
		categorySum[j] = int((categorySum[j] * 100) + 0.5) / 100.0	#round percentage
		results[j].append(categorySum[j])
	
out = pd.DataFrame({'Politics' : results[0],
					'Business' : results[1],
					'Technology' : results[2],
					'Film' : results[3],
					'Football' : results[4]},
					index = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])
out.to_csv('clustering_KMeans.csv', sep = "\t")

sys.stdout.write("\nResults stored to ./clustering_KMeans.csv\n")
sys.stdout.flush()
