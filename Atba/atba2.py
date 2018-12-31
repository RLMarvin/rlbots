import math, numpy as np
from rlbot.agents.base_agent import BaseAgent

U = 32768


class Atba2(BaseAgent):

    def initialize_agent(self):
        self.jump = 0
        self.timejump = 0
        self.jumpheight = 0
        self.dodge = 0
        self.timedodge = 0
        self.j = 0

    def get_output(self, packet):

        game = self.convert_packet_to_v3(packet)

        pIndex = self.index
        player = game.gamecars[pIndex] # 0 for p1, 1 for p2
        ball = game.gameball
        time = game.gameInfo.TimeSeconds

        ball_Location = _predictive(player,ball.Location,ball.Velocity) 

        dist = math.sqrt((player.Location.X - ball.Location.X)**2 + (player.Location.Y - ball.Location.Y)**2 + (player.Location.Z - ball.Location.Z)**2 )
        vel = abs(player.Velocity.X - ball.Velocity.X) + abs(player.Velocity.Y - ball.Velocity.Y) + abs(player.Velocity.Z- ball.Velocity.Y)
        x,y,z = local(ball_Location,[0,0,0],player.Rotation)
        _px,_py,_pz = local(player.Velocity,[0,0,0],player.Rotation)
        _bx,_by,_bz = local(ball.Velocity,[0,0,0],player.Rotation)
        _x,_y,_z = _px-_bx,_py-_by,_pz-_bz
        d,a,i = _spherical([x,y,z])
        r = _roll(player)
        _i,_r,_a = local(player.AngularVelocity,[0,0,0],player.Rotation)
        
        b_radius = 0.3048
        if z >210:
            b_radius = 0.6
        if z >450:
            b_radius = 0.7
        if z> 1200:
            b_radius = 0.5


        if (abs(a)>0.8 and _py>1000) or ( (z*b_radius)**2 > (d**2 - z**2) and abs(a) < 0.26 ) or (abs(a) > 0.75  and d < 700 ) or (d > z*2.8 and abs(a) < 0.03 and _y>1500 and z>500 and d<600) or (abs(a)>0.4 and _y>800 and abs(a)<0.75):
            Backwards = 32768
            Throttle = 0
            # if (d < 250 or d > 350) and abs(_y)<600 : a*=-1
        else : 
            Throttle = 32768
            Backwards = 0

        if abs(a) < 0.24 and Throttle > 32000 and _py < 2290 and d > 140  and (abs(0.5-abs(i))>0.3 or (player.Location.Z<25 and _y<0)) :
            Boost = 1
        else :
            Boost = 0

        if abs(a) > 0.46 and abs(a) < 0.8 and player.Location.Z < 20:
            powerslide = 1
        else:
            powerslide = 0

        h_i = 0.01*(0.5-abs(i))
        h_r = 0.05*(1-abs(r))

        if player.Location.Z>20:
            h_a = 0.05*(1-abs(a))
        else:
            h_a = 0.095*(1-abs(a))

        if (a - _a/11.25)*math.copysign(1,a) < 0 and abs(a)>0.01:
            if (player.Location.Z > 20 and on_wall(player.Location)==False) :
                a *= -1
        
        if (r - _r/22)*math.copysign(1,r) < 0 :
                r *= -1

        if (i + _i/23)*math.copysign(1,i) < 0 :
                i *= -1

        a_turn = (math.copysign( abs(a)**(1-abs(a)**h_a), a ) + 1 ) *16383
        i_turn = (math.copysign( abs(i)**(1-abs(i)**h_i), -i ) + 1 ) *16383
        r_turn = (math.copysign( abs(r)**(1-abs(r)**h_r), -r ) + 1 ) *16383

        if (abs(0.5-abs(a))>0.4 and abs(0.5-abs(i))>0.15 and abs(r)>0.3 and player.Location.Z>30 and on_wall(player.Location)==False 
            or ((25 < player.Location.Z < 250 and player.Velocity.Z < -400) or (player.Location.Z>1900 and player.Velocity.Z > 350)) and d > 400 and abs(r)>0.15 ): 
            # a_turn = r_turn
            # powerslide = 1
            0
        else:
            r_turn = 16383


        jump = 0

        if z > 155 and abs(a) < 0.05 and _y > -350 and d <z*(1.1+player.Boost/60) and z< 700 +player.Boost*14:
            Throttle = 0
            Backwards = 32768
            if self.jump == 0 and int(time*100)%2==0:
                self.jump = 1
                self.timejump = time
                # print("Double jump",int(time*10))

        # print(_py)
        if (d <= 220 ) and ( dist < 820 and vel > 0.600) and abs(z) <= 85 and abs(player.Location.Z-ball.Location.Z)< 100 and abs(0.5 -abs(i)) >0.15 :
            # print("Dodge Condition")
            if (player.Location.Z > 20 and self.dodge == 0) :
                # a_turn = 16383
                # if player.ball_Locationn.Z <40 :i_turn = 32768
                if (player.Location.Z >50 or (dist<110 and z<60)) :
                    # print("Dodging",int(time))
                    if self.dodge == 0 :
                        # jump = 1
                        self.dodge = 1
                        self.timedodge = time
            if  player.Location.Z < 20 or (on_wall(player.Location)==True and z>60) :
                if self.jump == 0 :
                    # print("self.jump")
                    self.jump = 1
                    self.timejump = time-0.19



        if (z>1350 or ((d<z*1.5 or vel<400) and player.Location.Z<500 and abs(a) < 0.15 and ball.Location.Z < 500)) and on_wall(player.Location)==True and player.Location.Z>60 and (abs(0.5-abs(a))>0.25 or d>2500) : #preventing the bot from staying on the walls
            if self.jump==0:
                self.jump = 1
                self.timejump = time
                # print('wall, jump')

        if ( (time > self.timejump +0.25 or z<20 ) and self.timejump !=0 and self.jump == 1  ):
            # print('reset')
            self.jump = 0
            self.timejump = time

        if ( time < self.timejump +0.05 ):
            a_turn = 16383
            i_turn = 16383

        if self.jump == 1 :
            if self.j == self.jump:
                jump = 1
        else: 
            jump = 0
        
        if player.Location.Z<250 and time < self.timedodge +abs(1-a/2)*0.66 and self.timedodge !=time and player.Location.Z > 34 :
            a_turn = 16383
            i_turn = 16383

        if time < self.timedodge +0.05 and self.timedodge !=time and player.Location.Z > 34 :
            i_turn =  abs(a)*32768
            a_turn = abs(Range180(a+0.5,1))*32768
            
        if self.dodge == 1 :
            self.jump=0
            jump = 1
            # print("Dodge",int(time))
            # print(a*180)  
            i_turn =  abs(a)*32768  
            a_turn = abs(Range180(a+0.5,1))*32768
            self.timedodge = time
            self.dodge = 0




        self.j = self.jump

        # return[int(a_turn), int(i_turn), Throttle, Backwards, jump, Boost, powerslide]
        output = [(Throttle-Backwards)/U,a_turn/(U/2)-1,i_turn/(U/2)-1,a_turn/(U/2)-1,r_turn/(U/2)-1,jump,Boost,powerslide]

        return self.convert_output_to_v4(output)










def xy_to_polar_coordinates(x,y): 
    d = math.sqrt(x*x+y*y)
    a = math.atan2(y,x)
    return d,a
def polar_to_xy_coordinates(d,a): 
    x = d*math.cos(a)
    y = d*math.sin(a)
    return

def xyz_to_spherical_coordinates(x,y,z): 
    d = math.sqrt(x*x+y*y+z*z)
    try : i = math.acos(z/d)
    except ZeroDivisionError: i=0
    a = math.atan2(y,x)
    return d,a,i
def spherical_to_xyz_coordinates(d,a,id): 
    x = d*math.sin(i)*math.cos(a)
    y = d*math.sin(i)*math.sin(a)
    z = d*cos(i)
    return x,y,z

def global_to_local_xy_coordinates(x,y,ox,oy,rot): 
    _x = (x-ox)*math.cos(rot) + (y-oy)*math.sin(rot)
    _y = -(x-ox)*math.sin(rot) + (y-oy)*math.cos(rot)
    return _x,_y
def global_to_local_xyz_coordinates(x,y,z,ox,oy,oz,pitch,yaw,roll): 

    r1,r2,r3 = pitch,roll,yaw

    Rx  = np.array([[      1,             0,               0      ],
                    [      0,        math.cos(r1),  -math.sin(r1) ],
                    [      0,        math.sin(r1),   math.cos(r1) ]])
    Ry  = np.array([[ math.cos(r2),       0,         math.sin(r2) ],
                    [      0,             1,               0      ],
                    [-math.sin(r2),       0,         math.cos(r2) ]])
    Rz  = np.array([[ math.cos(r3),  -math.sin(r3),        0      ],
                    [ math.sin(r3),   math.cos(r3),        0      ],
                    [      0,                 0,           1      ]])

    R   = np.dot(Rz,Rx)
    R   = np.dot(R,Ry)
    R   = R.T
    V   = np.dot(R,np.array([x-ox,y-oy,z-oz]))

    return V

def vector(V): 
    a = np.zeros(shape=(3))
    try: a[0]=V[0]
    except: 
        try: a[0]=V.X
        except: a[0]=V.Pitch
    try: a[1]=V[1]
    except: 
        try: a[1]=V.Y
        except: a[1]=V.Yaw
    try: a[2]=V[2]
    except: 
        try: a[2]=V.Z
        except: 
            try : a[2]=V.Roll
            except : print("Type Invalid")
    return a
def local(P,oP,R): 
    P = vector(P)
    oP = vector(oP)
    R = rotations(vector(R))
    x,y,z = global_to_local_xyz_coordinates(P[0],P[1],P[2],oP[0],oP[1],oP[2],R[0],R[1],R[2])
    return x,y,z
def _spherical(P): 
    P = vector(P)
    d,i,a = xyz_to_spherical_coordinates(P[0],P[1],P[2])
    return d, Range180(i-math.pi/2,math.pi)/math.pi, Range180(a-math.pi/2,math.pi)/math.pi
def rotations(R): 
    p = R[0]/32768*math.pi
    y = Range180(R[1]-16384,32768)/32768*math.pi
    r = - R[2]/32768*math.pi
    return p,y,r

def _roll(player):
    x,y,z = player.Location.X, player.Location.Y, player.Location.Z
    _x,_y,_z = player.Velocity.X, player.Velocity.Y, player.Velocity.Z
    x_,y_,z_ = x+_x/1.5, y+_y/1.5, z+_z/1.5
    r,a,i = player.Rotation.Roll/32768*180,player.Rotation.Yaw/32768*180,player.Rotation.Pitch/32768*180
    a90 = Range180(a+90,180)
    if z_ > 50  :
        if abs(x_) > 4000 and abs(x)<4050 and abs(y_)<5000   :
            r = Range180(r-math.copysign(90,a*x),180)
        if abs(y_) > 5000 and abs(x)<4000 and abs(90-abs(a))>30:
            r = Range180(r-math.copysign(90,-a90*y),180)
        if z_ > 1700 :
            r = math.copysign(180-abs(r),-r)
    return r/180

def Range180(value,pi): 
    return value - math.copysign( (pi*2) * (abs(value)//pi) ,value)
def pos(v):
    if v>0: return v
    else  : return 0

def on_wall(Location): 
    Location = vector(Location) 
    return (( abs(Location[0]) > 4000-100  or  abs(Location[1]) > 5000-100 or ( abs(Location[0]) > 2900-80 and abs(Location[1]) > 4000-80)) or Location[2] > 1800) and abs(Location[1]<5150 and abs(Location[0]) <4150)

def _predictive(player,target_loc,target_vel): 
    target_loc, target_vel = vector(target_loc), vector(target_vel)
    dist = math.sqrt((player.Location.X - target_loc[0])**2 + (player.Location.Y - target_loc[1])**2 + (player.Location.Z - target_loc[2])**2 )
    _loc = vector([ target_loc[0] - player.Location.X,  target_loc[1] - player.Location.Y,  target_loc[2] - player.Location.Z  ])
    
    g = 0
    if player.Location.Z > 25 and on_wall(player.Location)==False and dist>1500 and target_loc[2]-player.Location.Z>500:
        g = 42  
    if target_loc[2]-player.Location.Z>700 and player.Location.Z <20 and dist < 5500 and dist>1500:
        g = -200
    
    player_vel = vector([player.Velocity.X,player.Velocity.Y,player.Velocity.Z*1-g])
    
    _target = _loc + target_vel*1*(dist/2450) - player_vel*1*(dist/2450)

    if target_loc[2] + target_vel[2]*1*dist/2500<60:
        _target[2]=(_target[2]-60)*(-0.1) +60
        # _target[1]=_loc[1] + ((target_vel[1] - p_vel[1])*0.9*dist/2500)
        # _target[0]=_loc[0] + ((target_vel[0] - p_vel[0])*0.9*dist/2500)

    return _target