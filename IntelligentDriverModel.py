def rate_func( t, V ):
    # RATE_FUNC: IDM Car model
    # Model a car approaching a solid wall
    
    # unpack
    x = V[0] # position
    v = V[1] # velocity
    
    # Compute acceleration from IDM

    a_idm = 3 # what's the unit?
          
    # compute derivatives
    dx = v       #(from line 7)
    dv = a_idm   #(from line 11)
    
    # pack rate array
    rate = array([dx, dy])
    
    Vblock = v0/2
    yinit = 
    return rate

# set parameters
T = 1.8 #time headway
delta_exp = 4
L = 5
a_accel = 0.2
b_decel = 3
v0 = 28 #desired speed in m/s
s0 = 2.0 #desired gap m

Xblock = 5000. # front bumper of stopped car
Vblock = v0/2
# set initial conditions
xinit = 5
vinit = 10

# pack i.c.
X0 = array([yinit, vinit])

# set the time interval for solving
Tstart=0
Tend = 40

# Form Time array
time = linspace(Tstart,Tend,400) # 400 steps for nice plot