###### Write Your Library Here ###########

import heapq
import copy

class myNode:
    def __init__(self, route, location, unvisited_goals):
        self.route = copy.deepcopy(route)
        self.location=location
        self.unv_g = unvisited_goals
        self.obj=[]

        # F = G+H
        self.f=0
        self.g=0
        self.h=0

    def __eq__(self, other):
        return self.location==other.location and str(self.obj)==str(other.obj)

    def __le__(self, other):
        return self.g+self.h<=other.g+other.h

    def __lt__(self, other):
        return self.g+self.h<other.g+other.h

    def __gt__(self, other):
        return self.g+self.h>other.g+other.h

    def __ge__(self, other):
        return self.g+self.h>=other.g+other.h

class Edge:
    def __init__(self, i, j, w):
        self.v1 = i
        self.v2 = j
        self.weight = w
    def __eq__(self, other):
        return self.weight==other.weight and (self.v1, self.v2) == (other.v1, other.v2)

    def __le__(self, other):
        return self.weight<=other.weight

    def __lt__(self, other):
        return self.weight<other.weight

    def __gt__(self, other):
        return self.weight>other.weight

    def __ge__(self, other):
        return self.weight>=other.weight

def p1_to_p2(maze, p1, p2):
    startPoint = p1
    endPoint = p2
    route = {}
    visited = []
    path = []
    
    heap = []
    newNode = Node((-1,-1), startPoint)
    newNode.g = 0
    newNode.h = manhatten_dist(startPoint, endPoint)
    newNode.f = newNode.g + newNode.h
    heapq.heappush(heap, newNode)
    
    while(1):
        now = heapq.heappop(heap)
        route[now.location] = now.parent
        if (now.location == endPoint): break
        visited.append(now.location)
        for canGo in maze.neighborPoints(now.location[0], now.location[1]):
            if canGo in visited: continue
            newNode = Node(now.location, canGo)
            newNode.g = now.g + 1
            newNode.h = manhatten_dist(canGo, endPoint)
            newNode.f = newNode.g + newNode.h
            heapq.heappush(heap, newNode)
        
    tmp = now.location
    while (route[tmp] != (-1,-1)):
        path.insert(0,tmp)
        tmp = route[tmp]
    path.insert(0, tmp)

    return path

def smallerPath(route1, route2):
    if(len(route1) < len(route2)): return route1
    return route2

def unvisited_circles(end_points, unv, numV):
    Vlist = []
    for i in range(numV):
        if (1<<i) & unv == 0:
            Vlist.append(end_points[i])
    return Vlist

#########################################


def search(maze, func):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_four_circles": astar_four_circles,
        "astar_many_circles": astar_many_circles
    }.get(func)(maze)


# -------------------- Stage 01: One circle - BFS Algorithm ------------------------ #

def bfs(maze):
    
    """
    [문제 01] 제시된 stage1의 맵 세가지를 BFS Algorithm을 통해 최단 경로를 return하시오.(20점)
    """

    start_point=maze.startPoint()
    path=[]
    
    ####################### Write Your Code Here ################################
    
    end_points=maze.circlePoints()
    end_points.sort()
    goals=len(end_points) ## number of goals
    visited = []
    pIndex = [-1] ## save parent's index
    route = [(start_point,0)] ## save route
    pos = -1
    
    while(1):
        pos += 1
        if(route[pos][1] == goals): break
        
        for x in maze.neighborPoints(route[pos][0][0], route[pos][0][1]):
            if x in visited: continue
            pIndex.append(pos)
            visited.append(x)
            if(x in end_points): route.append((x, route[pos][1]+1))
            else: route.append((x, route[pos][1]))
    
    while (pos != -1):
        path.insert(0,route[pos][0])
        pos = pIndex[pos]

    return path

    ############################################################################



class Node:
    def __init__(self,parent,location):
        self.parent=parent
        self.location=location #현재 노드

        self.obj=[]

        # F = G+H
        self.f=0
        self.g=0
        self.h=0

    def __eq__(self, other):
        return self.location==other.location and str(self.obj)==str(other.obj)

    def __le__(self, other):
        return self.g+self.h<=other.g+other.h

    def __lt__(self, other):
        return self.g+self.h<other.g+other.h

    def __gt__(self, other):
        return self.g+self.h>other.g+other.h

    def __ge__(self, other):
        return self.g+self.h>=other.g+other.h


# -------------------- Stage 01: One circle - A* Algorithm ------------------------ #

def manhatten_dist(p1,p2):
    return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])

def astar(maze):

    """
    [문제 02] 제시된 stage1의 맵 세가지를 A* Algorithm을 통해 최단경로를 return하시오.(20점)
    (Heuristic Function은 위에서 정의한 manhatten_dist function을 사용할 것.)
    """

    start_point=maze.startPoint()
    end_point=maze.circlePoints()[0]

    path=[]

    ####################### Write Your Code Here ################################

    route = {} ## save route
    visited = []
    
    heap = []
    newNode = Node((-1,-1), start_point)
    newNode.g = 0
    newNode.h = manhatten_dist(start_point, end_point)
    newNode.f = newNode.g + newNode.h
    heapq.heappush(heap, newNode)
    
    while(1):
        now = heapq.heappop(heap)
        route[now.location] = now.parent
        if (now.location == end_point): break
        visited.append(now.location)
        for canGo in maze.neighborPoints(now.location[0], now.location[1]):
            if canGo in visited: continue
            newNode = Node(now.location, canGo)
            newNode.g = now.g + 1
            newNode.h = manhatten_dist(canGo, end_point)
            newNode.f = newNode.g + newNode.h
            heapq.heappush(heap, newNode)
        
    tmp = now.location
    while (route[tmp] != (-1,-1)):
        path.insert(0,tmp)
        tmp = route[tmp]
    path.insert(0, tmp)

    return path

    ############################################################################


# -------------------- Stage 02: Four circles - A* Algorithm  ------------------------ #



def stage2_heuristic(end_points, location, m_cycle):
    h = 1000000
    for x in range(4):
        h = min(h, manhatten_dist(end_points[x], location)+len(m_cycle[x]))
    return h


def astar_four_circles(maze):
    """
    [문제 03] 제시된 stage2의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage2_heuristic function을 직접 정의하여 사용해야 한다.)
    """

    end_points=maze.circlePoints()
    end_points.sort()

    path=[]
    ####################### Write Your Code Here ################################
    
    m_way = [[[] for i in range(4)] for j in range(4)] 
    for i in range(4):
        for j in range(0,4):
            if i != j: m_way[i][j] = p1_to_p2(maze, end_points[i], end_points[j])
    m_cycle = [[] for i in range(4)]
    for i in range(4):
        m_cycle[i] = m_way[i%4][(i+1)%4] + m_way[(i+1)%4][(i+2)%4][1::] + m_way[(i+2)%4][(i+3)%4][1::]
        m_cycle[i] = smallerPath(m_cycle[i],m_way[i%4][(i+1)%4]+m_way[(i+1)%4][(i+3)%4][1::]+m_way[(i+3)%4][(i+2)%4][1::])
        m_cycle[i] = smallerPath(m_cycle[i],m_way[i%4][(i+2)%4]+m_way[(i+2)%4][(i+1)%4][1::]+m_way[(i+1)%4][(i+3)%4][1::])
        m_cycle[i] = smallerPath(m_cycle[i],m_way[i%4][(i+2)%4]+m_way[(i+2)%4][(i+3)%4][1::]+m_way[(i+3)%4][(i+1)%4][1::])
        m_cycle[i] = smallerPath(m_cycle[i],m_way[i%4][(i+3)%4]+m_way[(i+3)%4][(i+1)%4][1::]+m_way[(i+1)%4][(i+2)%4][1::])
        m_cycle[i] = smallerPath(m_cycle[i],m_way[i%4][(i+3)%4]+m_way[(i+3)%4][(i+2)%4][1::]+m_way[(i+2)%4][(i+1)%4][1::])
    heap = []
    newNode = myNode([maze.startPoint()], maze.startPoint(),0)
    newNode.g = 0
    newNode.h = stage2_heuristic(end_points, newNode.location, m_cycle)
    newNode.f = newNode.g + newNode.h
    heapq.heappush(heap, newNode)
    
    while(1):
        now = heapq.heappop(heap)
        if(now.location in end_points):
            t = end_points.index(now.location)
            break
        for canGo in maze.neighborPoints(now.location[0], now.location[1]):
            newNode = myNode(now.route + [canGo], canGo, 0)
            newNode.g = now.g+1
            newNode.h = stage2_heuristic(end_points, newNode.location, m_cycle)
            newNode.f = newNode.g + newNode.h
            heapq.heappush(heap, newNode)    

    path = copy.deepcopy(now.route)
    del path[len(path)-1]
    path += m_cycle[t]
    return path

    ############################################################################



# -------------------- Stage 03: Many circles - A* Algorithm -------------------- #

def mst(verteces, edges, numV, setV):

    cost_sum=0
    ####################### Write Your Code Here ################################
    connected = []
    if cache[verteces] != -1: return cache[verteces]
    if edges == []: return 0
    while 1:
        if (connected != []) and (setV == connected[0]):break
        e = heapq.heappop(edges)
        f1 = False
        f2 = False

        for x in connected:
            if e.v1 in x:
                f1 = True
                break
        for y in connected:
            if e.v2 in y:
                f2 = True
                break
        if f1:
            if f2:
                if x == y: continue
                connected.append(x|y)
                connected.remove(x)
                connected.remove(y)
            else:
                x.add(e.v2)
        else:
            if f2:
                y.add(e.v1)
            else:
                connected.append({e.v1, e.v2})
        cost_sum += e.weight
    cache[verteces] = cost_sum

    return cost_sum

    ############################################################################


def stage3_heuristic(node, edges, numV):
    gE = []
    setV = set()
    
    for edge in edges:
        if ((1 << edge.v1) & node.unv_g)==0 and ((1 << edge.v2) & node.unv_g)==0:
            heapq.heappush(gE, edge)
        if (2**edge.v1 & node.unv_g)==0: setV = setV | {edge.v1}
        if (2**edge.v2 & node.unv_g)==0: setV = setV | {edge.v2}

    return mst(node.unv_g, gE, numV, setV)


def astar_many_circles(maze):
    """
    [문제 04] 제시된 stage3의 맵 세가지를 A* Algorithm을 통해 최단 경로를 return하시오.(30점)
    (단 Heurstic Function은 위의 stage3_heuristic function을 직접 정의하여 사용해야 하고, minimum spanning tree
    알고리즘을 활용한 heuristic function이어야 한다.)
    """

    end_points= maze.circlePoints()
    end_points.sort()

    path=[]

    ####################### Write Your Code Here ################################
    numV = len(end_points)
    edges = []
    for i in range(numV):
        for j in range(numV):
            if i < j:
                heapq.heappush(edges, Edge(i,j,len(p1_to_p2(maze,end_points[i],end_points[j]))))
                
    heap = []
    newNode = myNode([maze.startPoint()], maze.startPoint(), 0)
    newNode.g = 0
    newNode.h = 0
    newNode.f = 0
    heapq.heappush(heap, newNode)
    
    global cache
    cache = dict()
    for i in range(2**numV):
        cache[i] = -1
        
    while(1):
        now = heapq.heappop(heap)
        if now.unv_g == 2**(numV)-1:
            break;
        for canGo in unvisited_circles(end_points,now.unv_g,numV):
            temp = p1_to_p2(maze, now.location, canGo)
            newNode = myNode(now.route + temp[1::], canGo, now.unv_g | (1 << end_points.index(canGo)))
            newNode.g = len(temp)-1 + now.g
            newNode.h = stage3_heuristic(newNode, edges, numV)
            newNode.f = newNode.g + newNode.h
            heapq.heappush(heap, newNode)
    path = now.route
    return path

    ############################################################################
