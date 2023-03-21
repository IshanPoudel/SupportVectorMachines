#Load the csv file and get the value.
#All numbers , 
#Last line is the output as alaways. 

def load_csv(filename):
    return

def euclidean_distance(row1 , row2):
    # Row 1 and row2 are lists and contain the requored data
    distance = 0.0
    for i in range (len(row1)-1):
        distance = distance + (row1[i]-row2[i])**2
    return distance

 

def get_neighbours(train , test_row , num_neighbors):
    all_dist=[]
    for rows in train:
        distance = euclidean_distance(rows , test_row)
        # The dictionary contains the train row and the value.
        all_dist.append((rows ,distance))
    #Once sorted , sort the array basedby distance
    all_dist.sort(key=lambda tup: tup[1])

    #Get the neighbors
    neighbors=list()
    for i in range(num_neighbors):
        neighbors.append(all_dist[i][0])
    return neighbors



dataset= [[2 ,3, 5 , 0] , 
          [3 ,5 , 7, 0] , 
          [3 , 5 ,6 , 0],
          [1 , 2, 3, 1],
          [-1 , 8 , 8 , 1]]

neighbors = get_neighbours(dataset , dataset[0] , 2)


def predict(train , test_row , num_neighbors):
    neighbors = get_neighbours(train , test_row , num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values) , key=output_values.count)
    return prediction

prediction = predict(dataset, dataset[0], 3)
print('Expected %d, Got %d.' % (dataset[0][-1], prediction))