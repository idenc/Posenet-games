import pygame,sys,math,pickle
from pygame.locals import *

#de casteljau algorithm python
#https://javascript.info/bezier-curve
#https://incolumitas.com/2013/10/06/plotting-bezier-curves/


pygame.init()

#Open Pygame window
screen = pygame.display.set_mode((640, 480),) #add RESIZABLE or FULLSCREEN
#title
pygame.display.set_caption("bezier curve")

BLACK=pygame.color.THECOLORS["black"]
WHITE=pygame.color.THECOLORS["white"]
RED=pygame.color.THECOLORS["red"]
GREEN=pygame.color.THECOLORS["green"]
BLUE=pygame.color.THECOLORS["blue"]
BROWN=pygame.color.THECOLORS["brown"]
CLOCK=pygame.time.Clock()
SCREEN_WIDTH=640
SCREEN_HEIGHT=480

font=pygame.font.SysFont('Arial', 15)


def collide(point,rect):
    collided=0
    if point[0]>=rect[0] and point[0]<rect[0]+rect[2] \
    and point[1]>=rect[1] and point[1]<rect[1]+rect[3]:
        collided=1
    return collided


def draw_bezier_curvre(point1,point2,point3,max_draw_time=50000,show_drawing=True):
    
    # draw bezier curve with 3 control points
    # by using De Casteljau’s algorithm 

  
    #t like "time"
    t=0
    t_end=1
    step=0.005
    draw_time=0
    curve=[]
    
    while t <= t_end:
      if draw_time>max_draw_time:
       draw_time=0
       ratio=t/t_end
       
       segment1_point=[point1[0]+(point2[0]-point1[0])*ratio,
                       point1[1]+(point2[1]-point1[1])*ratio]
       segment2_point=[point2[0]+(point3[0]-point2[0])*ratio,
                       point2[1]+(point3[1]-point2[1])*ratio]

       #moving point on the drawing segement
       drawing_point=[int(segment1_point[0]+(segment2_point[0]-segment1_point[0])*ratio),
                       int(segment1_point[1]+(segment2_point[1]-segment1_point[1])*ratio)]

       curve.append(drawing_point)
       if show_drawing:
           pygame.draw.line(screen, BROWN, point1, point2, 1)
           pygame.draw.line(screen, BROWN, point2, point3, 1)
           pygame.draw.circle(screen,BROWN,point1,5,1)   
           pygame.draw.circle(screen,BROWN,point2,5,1)
           pygame.draw.circle(screen,BROWN,point3,5,1)
           text=font.render("1", True, (250,250,250))
           screen.blit(text,(point1[0]-10,point1[1]))
           text=font.render("2", True, (250,250,250))
           screen.blit(text,(point2[0]-10,point2[1]))
           text=font.render("3", True, (250,250,250))
           screen.blit(text,(point3[0]-10,point3[1]))
           
           pygame.draw.line(screen, BLUE, segment1_point, segment2_point, 2)
           pygame.draw.circle(screen,RED,drawing_point,5)
           
           for point in curve:
               pygame.draw.line(screen, WHITE, point, point, 2)

           t+=step
           pygame.display.flip()
           screen.fill(BLACK)
       t+=step
      draw_time+=1
      
      #if segment1_point>=point2:break
      
    return curve



def draw_bezier_curvre2(point1,point2,point3,point4,max_draw_time=50000,show_drawing=True):
    
    # draw bezier curve with 4 control points
    # by using De Casteljau’s algorithm 


    #t like "time"
    t=0
    t_end=1
    step=1/230#0.005
    draw_time=0
    curve=[]
    
    while t <= t_end:
      if draw_time>max_draw_time:
       draw_time=0
       ratio=t/t_end

       #the two moving segement
       segment1_point=[point1[0]+(point2[0]-point1[0])*ratio,
                       point1[1]+(point2[1]-point1[1])*ratio]
       segment2_point=[point2[0]+(point3[0]-point2[0])*ratio,
                       point2[1]+(point3[1]-point2[1])*ratio]
       segment3_point=[point3[0]+(point4[0]-point3[0])*ratio,
                       point3[1]+(point4[1]-point3[1])*ratio]

       #the drawing segement
       segment4_point=[segment1_point[0]+(segment2_point[0]-segment1_point[0])*ratio,
                       segment1_point[1]+(segment2_point[1]-segment1_point[1])*ratio]
       segment5_point=[segment2_point[0]+(segment3_point[0]-segment2_point[0])*ratio,
                       segment2_point[1]+(segment3_point[1]-segment2_point[1])*ratio]       

       #moving point on the drawing segement
       drawing_point=[int(segment4_point[0]+(segment5_point[0]-segment4_point[0])*ratio),
                       int(segment4_point[1]+(segment5_point[1]-segment4_point[1])*ratio)]
       
       curve.append(drawing_point)

       if show_drawing:
           pygame.draw.line(screen, BROWN, point1, point2, 1)
           pygame.draw.line(screen, BROWN, point2, point3, 1)
           pygame.draw.line(screen, BROWN, point3, point4, 1)
           pygame.draw.circle(screen,BROWN,point1,5,1)   
           pygame.draw.circle(screen,BROWN,point2,5,1)
           pygame.draw.circle(screen,BROWN,point3,5,1)
           pygame.draw.circle(screen,BROWN,point4,5,1)
           text=font.render("1", True, (250,250,250))
           screen.blit(text,(point1[0]-10,point1[1]))
           text=font.render("2", True, (250,250,250))
           screen.blit(text,(point2[0]-10,point2[1]))
           text=font.render("3", True, (250,250,250))
           screen.blit(text,(point3[0]-10,point3[1]))
           text=font.render("4", True, (250,250,250))
           screen.blit(text,(point4[0]-10,point4[1]))
          
           pygame.draw.line(screen, GREEN, segment1_point, segment2_point, 2)
           pygame.draw.line(screen, GREEN, segment2_point, segment3_point, 2)
           pygame.draw.line(screen, BLUE, segment4_point, segment5_point, 2)
           pygame.draw.circle(screen,RED,drawing_point,5)

           for point in curve:
               pygame.draw.line(screen, WHITE, point, point, 2)

           pygame.display.flip()
           screen.fill(BLACK)
       t+=step
      draw_time+=1
    
    return curve



def draw_subsegments(points,ratio,curve,show_drawing):

        # a recursive function that use De Casteljau’s algorithm to draw all subsegements
        # this function is to use with draw_bezier_curvre3 function
        
        sub_points=[]
        
        for i in range(len(points)):
            
            if i< len(points)-1:
               j=1
            else:
               j=0

            if i< len(points)-2:
               segment1_point=[points[i][0]+(points[i+j][0]-points[i][0])*ratio,
                              points[i][1]+(points[i+j][1]-points[i][1])*ratio]
               segment2_point=[points[i+j][0]+(points[i+j+1][0]-points[i+j][0])*ratio,
                              points[i+j][1]+(points[i+j+1][1]-points[i+j][1])*ratio]
               
               if not len(sub_points):
                  sub_points.append(segment1_point)
               sub_points.append(segment2_point)

                  
               if show_drawing:
                  pygame.draw.line(screen, GREEN, segment1_point, segment2_point, 2)
               
               if len(points)==3:
                   
                  drawing_point=[int(segment1_point[0]+(segment2_point[0]-segment1_point[0])*ratio),
                                 int(segment1_point[1]+(segment2_point[1]-segment1_point[1])*ratio)]
                  
                  curve.append(drawing_point)
                  
                  if show_drawing:
                     pygame.draw.line(screen, BLUE, segment1_point, segment2_point, 2)
                     pygame.draw.circle(screen,RED,drawing_point,5)
                  
                     for point in curve:
                         pygame.draw.line(screen, WHITE, point, point, 2)
        
        if len(sub_points)>1:
           draw_subsegments(sub_points,ratio,curve,show_drawing)
          


def draw_bezier_curvre3(points,max_draw_time=50000,show_drawing=True):

    # draw bezier curve with any number of control points
    # by using a recursive De Casteljau’s algorithm 
           

    #t like "time"
    t=0
    t_end=1
    step=0.005
    draw_time=0
    curve=[]
    
    while t <= t_end:
      if draw_time>max_draw_time:
         draw_time=0
         ratio=t/t_end

         for i in range(len(points)):
           if i< len(points)-1:
             j=1
           else:
             j=0

           if show_drawing:
              pygame.draw.line(screen, BROWN, points[i], points[i+j], 1)
              pygame.draw.circle(screen,BROWN,points[i],5,1)   
              text=font.render(str(i+1), True, (250,250,250))
              screen.blit(text,(points[i][0]-10,points[i][1]))
           
           draw_subsegments(points,ratio,curve,show_drawing)
           
         if show_drawing:  
            pygame.display.flip()
            screen.fill(BLACK)
         t+=step
      draw_time+=1
      
    return curve

    

point1=[320,480]
point2=[320,400]
point3=[320,300]
point4=[320,250]
points_list=[[point1,point2,point3,point4]]
rects_list=[]

for points in points_list:
    rects=[]
    for point in points:
        rect=[point[0]-5,point[1]-5,10,10]
        rects.append(rect)
    rects_list.append(rects)

draw_curve=True
selected_point=None
selected_rect_index=None
selected_rect=None
show_drawing=False
max_draw_time=0#50000
move_speed=10
curves=[]
horizon_y=SCREEN_HEIGHT//2+10
center_x=SCREEN_WIDTH//2
#to exit double for loop
to_break=False


try:
    with open('curve.road', 'rb') as file:
         file_loader=pickle.Unpickler(file)
         #we don't need this data so we delet it
         junk=file_loader.load()
         del(junk)
         #but this we need it
         points_list=file_loader.load()
         rects_list=file_loader.load()
except FileNotFoundError: pass


pygame.key.set_repeat(400, 30)


while True:
    
    #loop speed limitation
    #30 frames per second is enough
    CLOCK.tick(30)
    
    for event in pygame.event.get():    #wait for events
        if event.type == QUIT:
            pygame.quit()
            sys.exit()


    if event.type == KEYDOWN:
        
       if event.key == K_UP:
          for i in range(len(points_list)):
              for j in range(len(points_list[i])):
                  points_list[i][j][1]-=move_speed
                  rects_list[i][j][1]-=move_speed
          for curve in curves:
              for point in curve:
                  point[1]-=move_speed
          draw_curve=True

         
       elif event.key == K_DOWN:
          for i in range(len(points_list)):
              for j in range(len(points_list[i])):
                  points_list[i][j][1]+=move_speed
                  rects_list[i][j][1]+=move_speed
          for curve in curves:
              for point in curve:
                  point[1]+=move_speed
          draw_curve=True
          

       if event.key == K_LEFT:
          for i in range(len(points_list)):
              for j in range(len(points_list[i])):
                  points_list[i][j][0]-=move_speed
                  rects_list[i][j][0]-=move_speed
          for curve in curves:
              for point in curve:
                  point[0]-=move_speed
          draw_curve=True

          
       elif event.key == K_RIGHT:
          for i in range(len(points_list)):
              for j in range(len(points_list[i])):
                  points_list[i][j][0]+=move_speed
                  rects_list[i][j][0]+=move_speed
          for curve in curves:
              for point in curve:
                  point[0]+=move_speed
          draw_curve=True


       if event.key == K_p:
          if len(curves):
             last_curve=curves[len(curves)-1]
             last_point=last_curve[len(last_curve)-1]
             point1=[last_point[0],last_point[1]]
             point2=[point1[0],point1[1]-50]
             point3=[point2[0],point2[1]-50]
             point4=[point3[0],point3[1]-50]
             points=[point1,point2,point3,point4]
             points_list.append(points)
             rects=[]
             for point in points:
                 rect=[point[0]-5,point[1]-5,10,10]
                 rects.append(rect)
             rects_list.append(rects)
             draw_curve=True
          else:
             point1=[320,480]
             point2=[320,400]
             point3=[320,300]
             point4=[320,250]
             points_list=[[point1,point2,point3,point4]]
             rects_list=[]
             for points in points_list:
                 rects=[]
                 for point in points:
                     rect=[point[0]-5,point[1]-5,10,10]
                     rects.append(rect)
                 rects_list.append(rects)
             draw_curve=True

       
       if event.key == K_c:
          points_list=[]
          rects_list=[]
          curves=[]
          draw_curve=True

          
       if event.key == K_f:
          first_point=points_list[0][0]
          if first_point!=[center_x,SCREEN_HEIGHT]:
             dx=(first_point[0]-center_x)
             dy=(first_point[1]-SCREEN_HEIGHT)
             for i in range(len(points_list)):
                 for j in range(len(points_list[i])):
                     point=points_list[i][j]
                     rect=rects_list[i][j]
                     point[0]-=dx
                     point[1]-=dy  
                     rect[0]-=dx
                     rect[1]-=dy
          draw_curve=True
          
           
       if event.key == K_RETURN:
          level=[]
          for curve in curves:
              for point in curve:
                  dx=point[0]-center_x
                  level.append(dx)
          with open('curve.road', 'wb') as file:
               file_saver=pickle.Pickler(file)
               file_saver.dump(level)
               file_saver.dump(points_list)
               file_saver.dump(rects_list)
          i=0
          while i<30:
              text=font.render("level saved", True, (250,250,250))
              screen.blit(text,(20,20))
              pygame.display.flip()
              i+=1
          draw_curve=True
               
          

    if event.type == MOUSEBUTTONDOWN:
       if event.button == 1:
          for i in range(len(rects_list)):
              for j in range(len(rects_list[i])):
                  if collide(event.pos,rects_list[i][j]):
                     selected_point=points_list[i][j]
                     selected_rect=rects_list[i][j]
       elif event.button == 3:
          for i in range(len(rects_list)):
              if to_break:
                 to_break=False
                 break
              for j in range(len(rects_list[i])):
                  if collide(event.pos,rects_list[i][j]):
                     del(points_list[i])
                     del(rects_list[i])
                     del(curves[i])
                     draw_curve=True
                     to_break=True
                     break
                     
    if selected_point:             
       if event.type == MOUSEBUTTONUP:
          selected_point[0]=event.pos[0]
          selected_point[1]=event.pos[1]
          selected_rect[0]=selected_point[0]-5
          selected_rect[1]=selected_point[1]-5
          selected_point=None
          draw_curve=True
       
       if event.type == MOUSEMOTION:
          selected_point[0]=event.pos[0]
          selected_point[1]=event.pos[1]
          selected_rect[0]=selected_point[0]-5
          selected_rect[1]=selected_point[1]-5
          if not show_drawing:
             draw_curve=True
             

                
    if  draw_curve:
        curves=[]
        for points in points_list:
            point1=points[0]
            point2=points[1]
            point3=points[2]
            point4=points[3]
            curve=draw_bezier_curvre2(point1,point2,point3,point4,max_draw_time,show_drawing)
            curves.append(curve)
        draw_curve=False


        for i in range(len(points_list)):
              for j in range(len(points_list[i])):
                   if j< len(points_list[i])-1:
                      k=1
                   else:
                      k=0
                   pygame.draw.line(screen, BROWN, points_list[i][j], points_list[i][j+k], 1)
                   pygame.draw.circle(screen,BROWN,points_list[i][j],5,1)   
                   text=font.render(str(j+1), True, (250,250,250))
                   screen.blit(text,(points_list[i][j][0]-10,points_list[i][j][1]))
                   #pygame.draw.rect(screen, (0,200,255), rects_list[i][j], 1)
                
        for curve in curves:       
            for point in curve:
                pygame.draw.line(screen, WHITE, point, point, 2)
        
        pygame.draw.line(screen, BLUE, (0,horizon_y), (SCREEN_WIDTH,horizon_y), 1)
        pygame.draw.line(screen, BLUE, (center_x,SCREEN_HEIGHT), (center_x,horizon_y), 1)
        
        pygame.display.flip()
        screen.fill(BLACK)
    #break
