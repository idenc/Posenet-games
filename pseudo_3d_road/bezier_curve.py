import pygame,sys,math
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
    step=0.005
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

    

point1=[150,380]
point2=[320,380]
point3=[320,100]
point4=[490,100]
point5=[490,240]
points=[point1,point2,point3,point4,point5]
rects=[]

for point in points:
    rect=[point[0]-5,point[1]-5,10,10]
    rects.append(rect)

draw_curve=True
selected_point=None
selected_rect_index=None
show_drawing=False
max_draw_time=0#50000
move_speed=10
curve=[]

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
          for i in range(len(points)):
            points[i][1]-=move_speed
            rects[i][1]-=move_speed
          for point in curve:
              point[1]-=move_speed
          draw_curve=True
          
       elif event.key == K_DOWN:
           for i in range(len(points)):
             points[i][1]+=move_speed
             rects[i][1]+=move_speed
           for point in curve:
              point[1]+=move_speed
           draw_curve=True   


       if event.key == K_LEFT:
          for i in range(len(points)):
            points[i][0]-=move_speed
            rects[i][0]-=move_speed
          for point in curve:
              point[0]-=move_speed
          draw_curve=True


       elif event.key == K_RIGHT:
          for i in range(len(points)):
            points[i][0]+=move_speed
            rects[i][0]+=move_speed
          for point in curve:
              point[0]+=move_speed
          draw_curve=True
          
           

    if event.type == MOUSEBUTTONDOWN:
       if event.button == 1:
          for i in range(len(rects)):
              if collide(event.pos,rects[i]):
                 selected_point=points[i]
                 selected_rect_index=i
                 
    if selected_point:             
       if event.type == MOUSEBUTTONUP:
          selected_point[0]=event.pos[0]
          selected_point[1]=event.pos[1]
          rects[selected_rect_index]=[selected_point[0]-5,selected_point[1]-5,10,10]
          selected_point=None
          draw_curve=True
       
       if event.type == MOUSEMOTION:
          selected_point[0]=event.pos[0]
          selected_point[1]=event.pos[1]
          rects[selected_rect_index]=[selected_point[0]-5,selected_point[1]-5,10,10]
          if not show_drawing:
             draw_curve=True
             
                
    if  draw_curve:   
        #curve=draw_bezier_curvre2(point1,point2,point3,point4,max_draw_time,show_drawing)
        curve=draw_bezier_curvre3(points,max_draw_time,show_drawing)
        draw_curve=False

        for i in range(len(points)):
               if i< len(points)-1:
                  j=1
               else:
                  j=0
               pygame.draw.line(screen, BROWN, points[i], points[i+j], 1)
               pygame.draw.circle(screen,BROWN,points[i],5,1)   
               text=font.render(str(i+1), True, (250,250,250))
               screen.blit(text,(points[i][0]-10,points[i][1]))
               #pygame.draw.rect(screen, (0,200,255), rects[i], 1)

               
        for point in curve:
            pygame.draw.line(screen, WHITE, point, point, 2)

        
        pygame.display.flip()
        screen.fill(BLACK)
    #break
