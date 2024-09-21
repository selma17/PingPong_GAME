
import numpy as np
import cv2
import random
import pygame
import sys
from pygame.locals import *

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change to 1 if your camera is at index 1

if not cap.isOpened():
    print("Error: Unable to open the camera")
    sys.exit()

# Pygame initialization
p1 = 100
p2 = 100
pygame.init()
fps = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
RED = (174, 74, 52)
GREEN = (131, 166, 151)
BLUE = (15, 157, 232) 
BLACK = (0, 0, 0)

# Globals
WIDTH = 600
HEIGHT = 400
BALL_RADIUS = 20
PAD_WIDTH = 8
PAD_HEIGHT = 80  # Height of both paddles (reduced for player 2)
HALF_PAD_WIDTH = PAD_WIDTH // 2
HALF_PAD_HEIGHT = PAD_HEIGHT // 2
ball_pos = [0, 0]
ball_vel = [0, 0]
paddle1_vel = 0
paddle2_vel = 0
l_score = 0
r_score = 0
best_score = 0  # Best score variable

# Load your logo
logo = pygame.image.load('logo.png')  # Replace with your logo file
logo = pygame.transform.scale(logo, (100, 100))  # Resize the logo as needed

# Canvas declaration
window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)  # Fullscreen mode
pygame.display.set_caption('Ping Pong')

# Helper function to spawn a ball
def ball_init(right):
    global ball_pos, ball_vel
    ball_pos = [WIDTH // 2, HEIGHT // 2]
    horz = random.randrange(3, 4)
    vert = random.randrange(2, 4)
    
    if not right:
        horz = -horz
    
    ball_vel = [horz, -vert]

# Initialize game state
def init():
    global paddle1_pos, paddle2_pos, l_score, r_score
    paddle1_pos = [HALF_PAD_WIDTH - 1, HEIGHT // 2]
    paddle2_pos = [WIDTH + 1 - HALF_PAD_WIDTH, HEIGHT // 2]
    l_score = 0
    r_score = 0
    ball_init(random.choice([True, False]))

# Draw function
def draw(canvas):
    global paddle1_pos, paddle2_pos, ball_pos, ball_vel, l_score, r_score, best_score
    
    # Set background to blue
    canvas.fill(BLUE)
    
    # Draw the table lines and circle
    pygame.draw.line(canvas, WHITE, [WIDTH // 2, 0], [WIDTH // 2, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [PAD_WIDTH, 0], [PAD_WIDTH, HEIGHT], 1)
    pygame.draw.line(canvas, WHITE, [WIDTH - PAD_WIDTH, 0], [WIDTH - PAD_WIDTH, HEIGHT], 1)
    pygame.draw.circle(canvas, WHITE, [WIDTH // 2, HEIGHT // 2], 70, 1)

    # Display logo at the center of the screen
    canvas.blit(logo, (WIDTH // 2 - 50, HEIGHT // 2 - 50))  # Center the logo
    
    # Update paddle positions
    paddle1_pos[1] += paddle1_vel
    paddle2_pos[1] += paddle2_vel

    # Limit paddle movement to the window
    if paddle1_pos[1] < HALF_PAD_HEIGHT:
        paddle1_pos[1] = HALF_PAD_HEIGHT
    if paddle1_pos[1] > HEIGHT - HALF_PAD_HEIGHT:
        paddle1_pos[1] = HEIGHT - HALF_PAD_HEIGHT
    if paddle2_pos[1] < HALF_PAD_HEIGHT:
        paddle2_pos[1] = HALF_PAD_HEIGHT
    if paddle2_pos[1] > HEIGHT - HALF_PAD_HEIGHT:
        paddle2_pos[1] = HEIGHT - HALF_PAD_HEIGHT

    # Update ball position
    ball_pos[0] += int(ball_vel[0])
    ball_pos[1] += int(ball_vel[1])

    # Draw paddles and ball
    pygame.draw.circle(canvas, RED, ball_pos, BALL_RADIUS, 0)
    pygame.draw.polygon(canvas, GREEN, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT],
                                        [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
    
    # Draw the reduced size paddle for player 2
    pygame.draw.polygon(canvas, GREEN, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT],
                                        [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT],
                                        [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)

    # Ball collision with top and bottom walls
    if int(ball_pos[1]) <= BALL_RADIUS or int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
        ball_vel[1] = -ball_vel[1]

    # Ball collision with paddles or gutter
    if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and paddle1_pos[1] - HALF_PAD_HEIGHT <= int(ball_pos[1]) <= paddle1_pos[1] + HALF_PAD_HEIGHT:
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.3
        ball_vel[1] *= 1.3
    elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
        r_score += 1
        ball_init(True)
    
    if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and paddle2_pos[1] - HALF_PAD_HEIGHT <= int(ball_pos[1]) <= paddle2_pos[1] + HALF_PAD_HEIGHT:
        ball_vel[0] = -ball_vel[0]
        ball_vel[0] *= 1.3
        ball_vel[1] *= 1.3
    elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:  # Fix: only increment l_score if ball passes paddle
        l_score += 1
        ball_init(False)

    # Update scores
    if max(l_score, r_score) > best_score:
        best_score = max(l_score, r_score)  # Update best score

    myfont = pygame.font.SysFont("Comic Sans MS", 20)
    label1 = myfont.render(f"Score {l_score}", 1, (255, 255, 0))
    canvas.blit(label1, (50, 20))

    label2 = myfont.render(f"Score {r_score}", 1, (255, 255, 0))
    canvas.blit(label2, (470, 20))

    # Display the best score
    best_score_label = myfont.render(f"Best Score: {best_score}", 1, (255, 255, 0))
    canvas.blit(best_score_label, (250, 20))

# Game initialization
init()

# Game loop
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        continue

    # Convert to HSV and process the image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 5)
    
    # Player 1 (orange) detection
    lower_orange = np.array([0, 150, 150])
    upper_orange = np.array([30, 255, 255])
    mask_orange = cv2.inRange(blur, lower_orange, upper_orange)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    M_orange = cv2.moments(mask_orange)
    
    # Player 2 (blue) detection
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    mask_blue = cv2.inRange(blur, lower_blue, upper_blue)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    M_blue = cv2.moments(mask_blue)

    # Paddle 1 (Player 1) movement based on orange object
    if M_orange["m00"] != 0:
        cX1 = int(M_orange["m10"] / M_orange["m00"])
        cY1 = int(M_orange["m01"] / M_orange["m00"])
    
        cv2.circle(frame, (cX1, cY1), 5, (255, 255, 255), -1)
        cv2.putText(frame, "Player 1", (cX1 - 25, cY1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        for c in contours_orange:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

        # Paddle movement
        if cY1 > p1:
            paddle1_vel = 3
        elif cY1 < p1:
            paddle1_vel = -3
        else:
            paddle1_vel = 0
        p1 = cY1

    # Paddle 2 (Player 2) movement based on blue object
    if M_blue["m00"] != 0:
        cX2 = int(M_blue["m10"] / M_blue["m00"])
        cY2 = int(M_blue["m01"] / M_blue["m00"])
    
        cv2.circle(frame, (cX2, cY2), 5, (255, 255, 255), -1)
        cv2.putText(frame, "Player 2", (cX2 - 25, cY2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        for c in contours_blue:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

        # Paddle movement
        if cY2 > p2:
            paddle2_vel = 3
        elif cY2 < p2:
            paddle2_vel = -3
        else:
            paddle2_vel = 0
        p2 = cY2

    # Process Pygame events
    for event in pygame.event.get():
        if event.type == QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    # Draw game elements
    draw(window)

    # Update display
    pygame.display.update()
    fps.tick(60)
