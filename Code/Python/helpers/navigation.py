
# Website to draw a map: https://www.pixilart.com/draw#

'''
Insert the following code in another process to initiate communication

import socket

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    while True:
        #The string to be sent should be in the format: label + white space + x + white space + y
        #For example when retreiving shampoo: 2 0 1.3
        try:
            str1 = input("Enter label and coordinates: ")
            s.sendall(str1.encode())
        except:
            break
'''

from PIL import Image
import dijkstra as d
import math
import socket
import json

# Convert real coordinates to map coordinates
def convertRealToMap(scale, coordinate):
    # scale in meters/pixel
    return math.floor(coordinate/scale)

def rotate_around_point(coordinates, degrees, offset):
    x, y = coordinates
    offset_x, offset_y = offset
    cos_rad = math.cos(math.radians(degrees))
    sin_rad = math.sin(math.radians(degrees))
    qx = offset_x + cos_rad * x + sin_rad * y
    qy = offset_y + -sin_rad * x + cos_rad * y
    return qx, qy

def navigate(map_file, scale, entry_point, items_file, labels_file):
    # Each item "x" will have a file x_output.png which contains an image of the path
    output_file = "output.png"
    # Black
    COLOR_WALL = (0, 0, 0, 255)
    # White
    COLOR_PATH = (255, 255, 255, 255)
    # Red
    COLOR_START = (255, 0, 0, 255)
    # Green
    COLOR_END = (0, 255, 0, 255)
    # Blue
    COLOR_SOLVED = (0, 0, 255, 255)

    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)

    PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

    try:
        image = Image.open(map_file)
        imagePixels = image.load()
    except:
        print("Could not load file", map_file)
        exit()

    # Saving the imagePixels in a 2D array to refer to this whenever resetting pixels
    defaultPixels = [[None for _ in range(image.height)] for __ in range(image.width)]
    for x in range(image.width):
        for y in range(image.height):
            defaultPixels[x][y] = imagePixels[x, y]

    f = open(items_file)
    data = json.load(f)
    destinations = dict()
    distances = dict()
    graphs = dict()
    for k, v in data.items():
        destinations[k] = tuple(v)
        distances[k] = float("inf")
        graphs[k] = None

    f = open(labels_file)
    data = json.load(f)
    labels = dict()
    for k, v in data.items():
        labels[k] = tuple([tuple(v[0]), v[1]])

    # # access points of items
    # destinations = {
    #     "apple": (1.1, 0),
    #     "shampoo": (2.2, 1.4),
    #     "deodorant": (1.7, 0.5),
    # }
    # # Distances to reach the items from varying starting points
    # # Every time we reach an item, we remove it from the distances & graphs dictionary and set  visited flag in destinations as false
    # distances = {
    #     "apple": float("inf"),
    #     "shampoo": float("inf"),
    #     "deodorant": float("inf")
    # }
    # graphs = {
    #     "apple": None,
    #     "shampoo": None,
    #     "deodorant": None
    # }
    # # Label with its centre and rotation
    # labels = {
    #     "1": ((1.4, 1.7), 0),   # located in pixel (4,5) facing downward
    #     "2": ((1.7, 2.2), -90),  # located in pixel (5,7) facing the right
    #     "3": ((2.3, 1.3), -90)  # located in pixel (7,4) facing the right
    # }

    # Initial point that is replaced at the end with whatever destination we reach
    initialX = entry_point[0]
    initialY = entry_point[1]

    # Establishing socket and listening
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        print("Waiting for connection...")
        s.listen()
        conn, addr = s.accept()
        print("Connected")
        with conn:
            while len(distances) > 0:
                nodes = [[None for _ in range(image.width)] for __ in range(image.height)]
                for k, v in destinations.items():
                    # Resetting pixels to default pixel values every iteration
                    pixels = [[None for _ in range(image.height)] for __ in range(image.width)]
                    for x in range(image.width):
                        for y in range(image.height):
                            pixels[x][y] = defaultPixels[x][y]
                    pixels[convertRealToMap(scale, initialX)][convertRealToMap(scale, initialY)] = COLOR_START
                    pixels[convertRealToMap(scale, v[0])][convertRealToMap(scale, v[1])] = COLOR_END

                    for x in range(image.width):
                        for y in range(image.height):
                            pixel = pixels[x][y]
                            if pixel == COLOR_WALL:
                                nodes[y][x] = None
                            else:
                                nodes[y][x] = d.Node(x, y)
                            if pixel == COLOR_START:
                                initial_coords = (x, y)
                            if pixel == COLOR_END:
                                destination_coords = (x, y)

                    graph = d.Graph(nodes, initial_coords, destination_coords)
                    # Saving destination graph
                    graphs[k] = graph
                    # Saving destination distance
                    distances[k] = d.dijkstra(graph)

                # Retrieving item with minimum distance and plotting its path
                item = min(distances, key=distances.get)
                print("Item that we are going to retrieve: " + item)
                graph = graphs[item]

                initial_node = graph.graph[convertRealToMap(scale, initialY)][convertRealToMap(scale, initialX)]
                destination_node = graph.graph[convertRealToMap(scale, destinations[item][1])][convertRealToMap(scale, destinations[item][0])]

                nodes = graph.get_nodes()

                for node in nodes:
                    if node:
                        node.visited = False

                current_node = destination_node
                smallest_tentative_distance = distances[item]
                # Go from destination node to initial node to find path
                while current_node is not initial_node:
                    neighbors = graph.get_neighbors(current_node)
                    for neighbor in neighbors:
                        if not neighbor or neighbor.visited:
                            continue
                        if neighbor.tentative_distance < smallest_tentative_distance:
                            smallest_tentative_distance = neighbor.tentative_distance
                            neighbor.visited = True
                            current_node = neighbor
                    imagePixels[current_node.x, current_node.y] = COLOR_SOLVED

                imagePixels[destination_node.x, destination_node.y] = COLOR_END
                image.save(item + "_" + output_file, "PNG")
                # Resetting the image pixels that were coloured for the next item's path
                for x in range(image.width):
                    for y in range(image.height):
                        if imagePixels[x, y] == COLOR_SOLVED:
                            imagePixels[x, y] = COLOR_PATH

                currentRealX = currentRealY = -1
                # Giving directions until destination is reached
                while convertRealToMap(scale, currentRealX) != convertRealToMap(scale, destinations[item][0]) or convertRealToMap(scale, currentRealY) != convertRealToMap(scale, destinations[item][1]):
                    # Receiving coordinates from received from another process through socket
                    data = conn.recv(1024)
                    if not data:
                        raise Exception("Disconnected from client")
                    # Function that converts camera plane coordinates to map plane coordinates
                    labelNo, cameraRealX, cameraRealY = data.decode().split()
                    # Handling invalid label received
                    try:
                        currentRealX, currentRealY = rotate_around_point((float(cameraRealX), float(cameraRealY)), labels[labelNo][1], labels[labelNo][0])
                    except KeyError:
                        continue
                    currentMapX = convertRealToMap(scale, currentRealX)
                    currentMapY = convertRealToMap(scale, currentRealY)

                    for node in nodes:
                        if node:
                            node.visited = False

                    current_node = destination_node
                    smallest_tentative_distance = distances[item]
                    # Flag if directions are sent
                    directions_sent = False
                    # Go from destination node to initial node to find path
                    while current_node is not initial_node:
                        neighbors = graph.get_neighbors(current_node)
                        for neighbor in neighbors:
                            if not neighbor or neighbor.visited:
                                continue
                            if neighbor.tentative_distance < smallest_tentative_distance:
                                smallest_tentative_distance = neighbor.tentative_distance
                                neighbor.visited = True
                                # Printing the instructions on how to move to reach destination
                                if neighbor.x == currentMapX and neighbor.y == currentMapY:
                                    # Printing directions
                                    # print("Heading towards " + item + ": Move by " + str(current_node.x - neighbor.x) + " x blocks & " + str(current_node.y - neighbor.y) + " y blocks")
                                    string_to_send = "Heading towards " + item + ": Move by " + str(
                                        current_node.x - neighbor.x) + " x blocks & " + str(
                                        current_node.y - neighbor.y) + " y blocks"
                                    conn.sendall(string_to_send.encode())
                                    directions_sent = True
                                current_node = neighbor
                    if not directions_sent:
                        # Do nothing is sent when either an item has just been retrieved or location received is invalid
                        conn.sendall(b"Do nothing")
                # Change initial point at the end to the destination point
                initialX = currentRealX
                initialY = currentRealY
                # Remove the item from the destinations, distances and graphs dictionaries
                del destinations[item]
                del distances[item]
                del graphs[item]

navigate("../Images/Navigation/map.png", 0.3, (0.2, 2.3), "../Images/Navigation/items.json", "../Images/Navigation/labels.json")