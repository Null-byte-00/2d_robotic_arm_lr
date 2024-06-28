import pygame
import math

pygame.init()

WIDTH  = 500
HEIGHT = 500
GRAVITY = 0.001
screen = pygame.display.set_mode((WIDTH, HEIGHT))

fps = 60
timer = pygame.time.Clock()


class ConstantJoint:
    def __init__(self, screen, pos, id) -> None:
        self.screen = screen 
        self.pos = pos
        self.id = id
        self.connection = None
    
    def connect(self, joint):
        self.connection = joint
    
    def draw(self):
        pygame.draw.circle(self.screen, "white", self.pos, 5)
        if self.connection:
            pygame.draw.line(screen, "white", self.connection.pos, self.pos, 3)


class MovingJoint(ConstantJoint):
    def __init__(self, screen, pos, id, speed=0.0, retention=0.99, bounce_stop=0.00005) -> None:
        super().__init__(screen, pos, id)
        self.speed = speed
        self.retention = retention
        self.bounce_stop = bounce_stop
        self.angle = None
        self.distance = None
    
    def update_position(self):
        connected = self.connection
        x = self.pos[0]
        y = self.pos[1]
        x_c = connected.pos[0]
        y_c = connected.pos[1]

        if self.angle == None:
            self.angle = math.atan2(y_c - y, x_c - x)
        
        if self.distance == None:
            self.distance = math.sqrt(((x - x_c)**2) + ((y - y_c)**2))

        self.pos = (x_c + self.distance * math.cos(self.angle), 
                    y_c + self.distance * math.sin(self.angle))
        self.angle += self.speed

        if math.cos(self.angle) < 0:
            self.speed = self.speed * self.retention - GRAVITY
        elif abs(self.speed) < self.bounce_stop:
            self.speed = 0
        else:
            self.speed = self.speed * self.retention + GRAVITY


const_joint = ConstantJoint(screen, (250, 150), 1)
moving_joint = MovingJoint(screen, (310, 210), 2)
moving_joint2 = MovingJoint(screen, (250, 150), 3,  speed=0.1)
const_joint.connect(moving_joint)
moving_joint.connect(const_joint)
moving_joint2.connect(moving_joint)


def main():
    run = True
    while run:
        timer.tick(fps)
        screen.fill('black')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        moving_joint.update_position()
        moving_joint2.connection = moving_joint
        moving_joint2.update_position()

        moving_joint.draw()
        const_joint.draw()
        moving_joint2.draw()
    
        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()