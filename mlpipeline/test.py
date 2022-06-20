beams = [[j for i in range(2)] for j in range(3)]
print(beams)
beams = [list(i) for i in zip(*beams)]

print(beams)