import math

def main():
	# f = './a99sb-disp_openMM.xml'
	f='./SJ403.xml'
	f_out = './SJ403_1.xml'
	# f_out = './a99sb-disp_openMM_1.xml'
	f_out = open(f_out, 'w')
	with open(f, 'r') as f:
		for line in f:
			if 'angle=' in line or 'phase1=' in line or 'phase2=' in line or 'phase3=' in line or 'phase4=' in line or 'phase5=' in line or 'phase6=' in line:
				sline = line.split()
				start = ''
				for e in list(line):
					if e == '\t':
						start += e
					else:
						break
				end = '\n'
				for i,comp in enumerate(sline):
					if 'angle=' in comp:
						angle = float(comp[7:-1])*math.pi/180
						sline[i] = 'angle="{}"'.format(angle)
					elif 'phase' in comp:
						angle = float(comp[8:-1])*math.pi/180
						sline[i] = comp[:8]+'{}"'.format(angle)
				sline = ' '.join(sline)
				l = start+sline+end
				f_out.write(l)
			else:
				f_out.write(line)
	f_out.close()



if __name__ == "__main__":
	main()