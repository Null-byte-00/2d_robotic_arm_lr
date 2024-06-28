import pygame

pygame.init()

WIDTH  = 1000
HEIGHT = 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))

fps = 60
timer = pygame.time.Clock()
gravity = 0.5
bounce_stop = 0.3


def draw_walls():
    left = pygame.draw.line(screen, "white", (0,0), (0, HEIGHT), 10)
    right = pygame.draw.line(screen, "white", (WIDTH,0), (WIDTH, HEIGHT), 10)
    top = pygame.draw.line(screen, "white", (0,0), (WIDTH, 0), 10)
    bottom = pygame.draw.line(screen, "white", (0,HEIGHT), (WIDTH, HEIGHT), 10)


class Ball:
    def __init__(self, screen, x_pos=50, y_pos=50, radius=30, color='white', mass=100, 
                 retention=0.9, x_speed=0, y_speed=0, id=1) -> None:
        self.screen = screen
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = radius
        self.color = color
        self.mass = mass
        self.retention = retention
        self.x_speed = x_speed 
        self.y_speed = y_speed
        self.id = id
    
    def draw(self):
        self.circle = pygame.draw.circle(self.screen, self.color, (self.x_pos, self.y_pos), self.radius)
    
    def check_gravity(self):
        if self.y_pos < HEIGHT - self.radius - 5:
            self.y_speed += 0.5
        else:
            if self.y_speed > bounce_stop:
                self.y_speed = self.y_speed * -1 * self.retention
            elif abs(self.y_speed) <= bounce_stop:
                self.y_speed = 0

    def select_pos(self, pos):
        self.x_pos = pos[0]
        self.y_pos = pos[1]
    
    def update_position(self):
        self.y_pos += self.y_speed
        self.x_pos += self.x_speed


ball1 = Ball(screen)

# loop
run = True
while run:
    timer.tick(fps)
    screen.fill('black')

    draw_walls()
    ball1.draw()
    ball1.check_gravity()
    ball1.update_position()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                ball1.select_pos(event.pos)
                ball1.y_speed = 0
    
    pygame.display.flip()

pygame.quit()