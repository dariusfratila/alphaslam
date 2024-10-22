import pygame


class Display(object):
    def __init__(self, W, H):
        pygame.init()

        self.W, self.H = W, H
        print(f'{self.W, self.H}, self.W, self.H')
        self.window = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Alpha SLAM")

    def paint(self, img):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit(0)

        surf = pygame.surfarray.pixels3d(self.window)
        surf[:, :, 0:3] = img.swapaxes(0, 1)

        pygame.display.update()
