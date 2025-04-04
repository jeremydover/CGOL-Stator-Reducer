# A stator reducer for spaceships and oscillators. I hope

from __future__ import print_function
import re
import sys
import time
import argparse
from ortools.sat.python import cp_model
from array import *

class screen_printer:
	TEST = 0
	NORMAL = 1
	DEBUG = 2
	
	def __init__(self,print_level):
		self.print_level = print_level
	
	def set_print_level(self,print_level):
		if print_level > 2:
			self.print_level = 2
		else:
			self.print_level = print_level
		
	def print(self,level,target):
		if level <= self.print_level:
			print(target)

class ObjectiveSolutionPrinter(cp_model.CpSolverSolutionCallback):
	
	def __init__(self,variables):
		cp_model.CpSolverSolutionCallback.__init__(self)
		self.__solution_count = 0
		self.__start_time = time.time()
		self.__variables = variables

	def on_solution_callback(self):
		current_time = time.time()
		obj = self.ObjectiveValue()
		P.print(p.NORMAL,'Solution %i, time = %0.2f s, objective = %i' %
              (self.__solution_count, current_time - self.__start_time, obj))
		self.print_RLE()
		self.__solution_count += 1

	def solution_count(self):
		return self.__solution_count

	def print_RLE(self):
		height = len(self.__variables)
		width = len(self.__variables[0])
		my_values = [[self.Value(self.__variables[i][j]) for j in range(width)] for i in range(height)]
		rle = array_to_RLE(my_values)
		P.print(P.NORMAL,rle)

def RLE_to_tuples(rle):
	# First get bounding box sizes from RLE
	x = re.search("x = (\d+)",rle)
	if x is None:
		raise Exception("Invalid RLE syntax: missing width")
	my_width = int(x.group(1))
	
	y = re.search("y = (\d+)",rle)
	if y is None:
		raise Exception("Invalid RLE syntax: missing height")
	my_height = int(y.group(1))
	
	# Now strip header line, and eliminate carriage returns
	cells = re.sub('^.*\n','',rle)
	cells = cells.replace('\n','')
	
	live_cells = []
	current_row = 0
	current_column = 0
	while cells != '!':
		x = re.search('^(\d*)(b|o|\$)',cells)
		if x is None:
			raise Exception("Invalid RLE syntax")
		if x.group(1) == '':
			n = 1
		else:
			n = int(x.group(1))
		if x.group(2) == 'b':
			current_column = current_column + n
		elif x.group(2) == 'o':
			for i in range(n):
				live_cells.append((current_row,current_column+i))
			current_column = current_column + n
		else:
			current_column = 0
			current_row = current_row + n
		cells = re.sub('^(\d)*(b|o|\$)','',cells)
	return live_cells, my_height, my_width

def array_to_RLE(variables):
	height = len(variables)
	width = len(variables[0])
	
	rle = ''
	content = ''
	for i in range(height):
		for j in range(width):
			content = content + str(variables[i][j])
		content = content + '\n'
		
	while content != '\n':
		if content[0] == '\n':
			base = '$'
		elif content[0] == '0':
			base = 'b'
		else:
			base = 'o'
		count = re.search('[^'+content[0]+']',content).start();
		rle = rle + str(count) + base
		content = content[count:]
	rle = rle + '!'
	return rle

def test_border(border):
	# Converts list of integers to string
	my_string = ''.join(map(str,border))
	# We only need a new row/column if there are three contiguous live cells on the bounding box border of the
	# previous pattern.
	my_return = 1 if my_string.find('111') != -1 else 0
	return my_return
	

def test_expansion(pattern):
	my_height = len(pattern)
	my_width = len(pattern[0])
	
	return [test_border([pattern[i][0] for i in range(my_height)]),           # left border
			test_border([pattern[i][my_width-1] for i in range(my_height)]),  # right border
			test_border([pattern[0][j] for j in range(my_width)]), 	          # top border
			test_border([pattern[my_height-1][j] for j in range(my_width)])]  # bottom border

def process_pattern(pattern,period,is_gun):
	# Process the pattern RLE
	live_cells,orig_height,orig_width = RLE_to_tuples(pattern)
	P.print(P.NORMAL,"Initial population: "+str(len(live_cells)))
	
	# Initialize the first period of the life array. We're going to use python to run up a period of the
	# object, in order to figure out what cells are rotor/stator/blank, and also how big our bounding box
	# really is, in case the original size does not bound every phase of the object.
	
	# Well, this is a pain. When I did the re-factor to allow for the pattern box to increase, I messed up how
	# I was doing gun processing. How they're the same:
	# Regardless of which way I'm processing, I want to expand the grid on which I'm processing if the pattern leaves.
	# If we're in osciallator mode, I also want to expand the bounding_box of the search space.
	# If we're in gun mode, I do *not* want to expand the bounding_box. I just want to record the added cells
	#   as being in the ship channel, so I do not enforce periodicity there.
	# So it looks like I need to keep track of search_size and pattern_size differently. And so we also need to keep track of
	# the current search space, since it may float in the full frame depending on gun mode.
	
	current_search_height = orig_height
	current_search_width = orig_width
	
	current_pattern_height = orig_height
	current_pattern_width = orig_width
	
	# Variables will track where the origin of the original frame is in the expanded search array, as well as the offset
	# of the upper corner of the search space
	vert_orig_offset = 0
	horz_orig_offset = 0
	vert_search_offset = 0
	horz_search_offset = 0
		
	# Our main structure is lifeArray, which will contain the state of each cell at each phase of the period, so it will be 
	# a 3D array, indexed by period, then row, then column. 
	life_array = [[[0 for j in range(current_pattern_width)] for i in range(current_pattern_height)]]
	for x in live_cells:
		life_array[0][x[0]][x[1]] = 1

	# At each phase, we will determine if our bounding box size needs to increase. The number of addition rows/columns needed
	# will be stored in the bounding_box_expansion array, where each element indicates the number of additional rows/column
	# at left, right, top, bottom. No entry should be more than 1. And note, we will never shrink, even if the pattern does.
	# We are simply using this to create a maximum.
	bounding_box_expansion = [[0,0,0,0]]
		
	# Now let's apply life rules to the array.
	# Note: to fix a super-painful-to-find bug in the gun case, we actually need to step an extra time beyond period
	# All we really need this for is to determine if there are extra cells in the ship channel beyond the frame (spoiler:
	# there can be). But it does not hurt to run the pattern an extra tick. 
	for p in range(1,period+1):
		expansion = test_expansion(life_array[p-1])
		bounding_box_expansion.append(expansion)
		
		side = ['left','right','top','bottom']
		for i in range(4):
			if expansion[i] > 0:
				P.print(P.DEBUG,'At period '+str(p)+': expanded pattern window by '+str(expansion[i])+' cell'+('s' if expansion[i] != 1 else '')+' at '+side[i])
		
		# Now we know how big the array needs to be for this phase. We definitely increase the pattern size, and the offset 
		# of the original_bounding_box
		current_pattern_height = current_pattern_height+expansion[2]+expansion[3]
		vert_orig_offset = vert_orig_offset + expansion[2]
		current_pattern_width = current_pattern_width+expansion[0]+expansion[1]
		horz_orig_offset = horz_orig_offset + expansion[0]
		
		# We increase the search space if we are not in gun mode.
		if is_gun:
			# The search space does not expand, but gets moved by the expansion. Re-base it, if we add cells to either 
			# top or left. Adding to bottom or right does not move things, just adds extra outside the search space.
			vert_search_offset = vert_search_offset + expansion[2]
			horz_search_offset = horz_search_offset + expansion[0]
		else:
			# The search space remains the whole grid, so we just expand
			current_search_height = current_pattern_height
			current_search_width = current_pattern_width
		
		# Set up cell array for this phase
		this_phase_life_array = [[0 for j in range(current_pattern_width)] for i in range(current_pattern_height)]
		
		# If this box expanded, we have the problem that all previous life_array matrices are offset. We will need them
		# all to be lined up eventually, so let's just fix it here.
		if expansion[0] > 0 or expansion[1] > 0:
			for prior_period in range(p):
				for i in range(len(life_array[prior_period])):
					if expansion[0] > 0:
						life_array[prior_period][i] = [0 for k in range(expansion[0])] + life_array[prior_period][i]
					if expansion[1] > 0:
						life_array[prior_period][i] = life_array[prior_period][i] + [0 for k in range(expansion[1])]
						
		if expansion[2] > 0 or expansion[3] > 0:
			for prior_period in range(p):
				if expansion[2] > 0:
					life_array[prior_period] = [[0 for k in range(current_pattern_width)] for k in range(expansion[2])] + life_array[prior_period]
				if expansion[3] > 0:
					life_array[prior_period] = life_array[prior_period] + [[0 for k in range(current_pattern_width)] for k in range(expansion[3])]
	
		for i in range(current_pattern_height):
			for j in range(current_pattern_width):
				neighborhood = [life_array[p-1][i+k][j+m] for k in range(-1,2) for m in range(-1,2) if i+k in range(current_pattern_height) and j+m in range(current_pattern_width)]
				if life_array[p-1][i][j] == 0:					
					this_phase_life_array[i][j] = 1 if sum(neighborhood) == 3 else 0
				else:
					this_phase_life_array[i][j] = 1 if sum(neighborhood) in range(3,5) else 0
		
		life_array.insert(p,this_phase_life_array)
	
	# Looks like this might be another place where gun mode and oscillator mode differ. In oscillator mode, the search space 
	# remains the entire pattern. Only in gun mode is there a difference (at this stage). In gun mode, the search space remains
	# the original bounding box of the pattern
	
	# Classify all of the cells in the search space
	stator = []
	blank = []
	rotor = []
	external_ship_channel = []
	# Cells in the search space are considered either stator, blank or rotor. Even in gun mode, any ship cells are classified
	# as rotor. Either the original pattern needs to be configured such that the ship stream is periodic, or the --shipchannel
	# option can be passed to ensure that periodicity is not enforced on these cells.
	for i in range(vert_search_offset,vert_search_offset+current_search_height):
		for j in range(horz_search_offset,horz_search_offset+current_search_width):
			my_values = [life_array[p][i][j] for p in range(period)]
			if len(set(my_values)) == 1:
				if my_values[0] == 0:
					blank.append((i,j))
				else:
					stator.append((i,j))
			else:
				rotor.append([i,j,my_values])
	
	# Now take care of any cells not in the current search space. This only occurs in the gun case, in which case these pattern
	# cells form the "external" ship channel.
	for i in range(current_pattern_height):
		for j in range(current_pattern_width):
			if i not in range(vert_search_offset,vert_search_offset+current_search_height) or j not in range(horz_search_offset,horz_search_offset+current_search_width):
				if sum([life_array[p][i][j] for p in range(period+1)]) > 0:
					external_ship_channel.append((i,j))

	### NOTE!
	# If you look carefully at the code here, we leave open the possibility that a pattern could have three contiguous
	# cells on the border at period p-1, but we do NOT expand the grid or otherwise account for this. This is deliberate!
	# If the original input is an oscillator with the claimed period, then this case cannot happen...notice we're not 
	# checking periodicity here. If it is a gun, then this most likely does happen, but we don't want the stream of 
	# spaceships to mess us up.

	return orig_height,orig_width,current_pattern_height,current_pattern_width,vert_orig_offset,horz_orig_offset,current_search_height,current_search_width,vert_search_offset,horz_search_offset,stator,blank,rotor,external_ship_channel
	
def main(pattern,period,left_adjust=0,right_adjust=0,top_adjust=0,bottom_adjust=0,preserve=[],forceblank=[],is_gun=False,internal_ship_channel=[],model_test=False):
	
	# Send the initial pattern processing off to not clutter the modelling
	orig_height,orig_width,pattern_height,pattern_width,vert_orig_offset,horz_orig_offset,search_height,search_width,vert_search_offset,horz_search_offset,stator,blank,rotor,external_ship_channel = process_pattern(pattern,period,is_gun)
	
	# There are various subregions of the ultimate model board we need to track
	# 1. The pattern box; the minimum required for the original pattern to evolve. For now this is based at (0,0), and has
	#    dimensions pattern_height and pattern_width. If search space adjustments are made, we will use
	#    vert_pattern_offset and horz_pattern_offset to track.
	# 2. The original bounding box; the original RLE as given. Its top corner is determined by vert_orig_offset and
	#    horz_orig_offset and has dimensions orig_height and orig_width.
	# 3. The current search space; where we search for a new stator. Its top corner is determined by vert_search_offset and
	#    horz_search_offset, with dimensions search_height and search_width
	# 3. Life cells; the set of cells that obey the life evolution rules...this will be given shortly, but ultimately, it will
	#    be based at (1,1), and will have dimensions life_height and life_width.
	# 4. Model cells; all cells in the model...also given shortly; will be (0,0) with dimensions model_height and model_width.
			
	# Note: preserve, internal ship channel and forceblank are tied to the original frame, so it needs to be offset in case process_pattern made changes
	preserve = {(x[0]+vert_orig_offset,x[1]+horz_orig_offset) for x in preserve}
	forceblank = {(x[0]+vert_orig_offset,x[1]+horz_orig_offset) for x in forceblank}
	internal_ship_channel = {(x[0]+vert_orig_offset,x[1]+horz_orig_offset) for x in internal_ship_channel}
	
	## At this point, all of the tracking structures: stator, blank, rotor, preserve, internal_ship_channel and external_ship_channel are tied to the PATTERN position and offset. We will track the pattern offset through our various evolutions, and then rebase everything once adjustments are complete.

	# If we are going to expand the grid with adjustments, we need to add the additional rows/columns here. If
	# we are shrinking the board, we still leave extra. That way if we exclude something that is to be preserved 
	# or contains rotor cells, we will return INFEASIBLE, which is the desired behavior.
	
	life_height = pattern_height + max(0,top_adjust) + max(0,bottom_adjust)
	life_width = pattern_width + max(0,left_adjust) + max(0,right_adjust)
	
	# These adjustments affect the placement of the pattern box, the original bounding box, and the current search space.
	# Pattern box - does not need to expand...since evolution is unchanged, just shifted.
	vert_pattern_offset = max(0,top_adjust)
	horz_pattern_offset = max(0,left_adjust)
	
	# Original bounding box - does not expand, just shifted
	vert_orig_offset = vert_orig_offset + max(0,top_adjust)
	horz_orig_offset = horz_orig_offset + max(0,left_adjust)
	
	# Search space - oh, this is complicated, because it depends on whether we're expanding or shrinking
	# We only adjust the offset of the search box if we are shrinking at the top or left, since shrinking the
	# search space does not mean we shrink the pattern space.
	#
	# For example, if we are trying to trim one row off the top, the pattern space stays the same, since we 
	# still want to track life rules to make sure we're not expanding into that area. So the search space needs
	# to move down a row.
	vert_search_offset = vert_search_offset + max(0,-top_adjust)
	horz_search_offset = horz_search_offset + max(0,-left_adjust)
	
	search_height = search_height + top_adjust + bottom_adjust
	search_width = search_width + left_adjust + right_adjust
		
	# OK, one final adjustment. The life grid determines the cells we care about. But, we also need to make sure the pattern does
	# does not try to grow outside. Which means the model needs to be one cell larger on every side.
	model_height = life_height + 2
	model_width = life_width + 2
	
	# And this moves *everything* down one and right one. It's a pain to do this separately, but I think it's much clearer this way. And good thing...no sizes change, just offsets. And we may not need some of these moving on, but better safe.
	
	# The life grid
	vert_life_offset = 1
	horz_life_offset = 1
	
	# The pattern grid
	vert_pattern_offset = vert_pattern_offset + 1
	horz_pattern_offset = horz_pattern_offset + 1
	
	# The original grid
	vert_orig_offset = vert_orig_offset + 1
	horz_orig_offset = horz_orig_offset + 1
	
	# The search grid
	vert_search_offset = vert_search_offset + 1
	horz_search_offset = horz_search_offset + 1

	# Now that we have done all of the coordinate adjustments, we can finally rebase all of our sets that track the
	# state of the evolving pattern. Remember that they are based on the pattern offsets at this point.
	rotor = [[x[0] + vert_pattern_offset,x[1] + horz_pattern_offset,x[2]] for x in rotor]
	stator = {(x[0] + vert_pattern_offset,x[1] + horz_pattern_offset) for x in stator}
	blank = {(x[0] + vert_pattern_offset,x[1] + horz_pattern_offset) for x in blank}
	preserve = {(x[0] + vert_pattern_offset,x[1] + horz_pattern_offset) for x in preserve}
	forceblank = {(x[0] + vert_pattern_offset,x[1] + horz_pattern_offset) for x in forceblank}
	internal_ship_channel = {(x[0] + vert_pattern_offset,x[1] + horz_pattern_offset) for x in internal_ship_channel}
	external_ship_channel = {(x[0] + vert_pattern_offset,x[1] + horz_pattern_offset) for x in external_ship_channel}
	ship_channel = internal_ship_channel | external_ship_channel
	
	P.print(P.DEBUG,"Original Corner: ("+str(vert_orig_offset)+","+str(horz_orig_offset)+") Size: "+str(orig_height)+" x "+str(orig_width))
	P.print(P.DEBUG,"Pattern Corner: ("+str(vert_pattern_offset)+","+str(horz_pattern_offset)+") Size: "+str(pattern_height)+" x "+str(pattern_width))
	P.print(P.DEBUG,"Search Corner: ("+str(vert_search_offset)+","+str(horz_search_offset)+") Size: "+str(search_height)+" x "+str(search_width))
	P.print(P.DEBUG,"Life Corner: ("+str(vert_life_offset)+","+str(horz_life_offset)+") Size: "+str(life_height)+" x "+str(life_width))
	
	## SET UP THE SAT MODEL ##
	model = cp_model.CpModel()
	# Creates the cell variables: int and Boolean
	# 1 indicates on, 0 off

	P.print(P.NORMAL,"Size of search grid: "+str(life_height)+" by "+str(life_width))
	P.print(P.TEST,"Size of input stator: "+str(len(stator)))

	# Set up variables
	a = []
	b = []
  
	for p in range(period):
		ai = []
		bi = []
		for i in range(model_height):
			aj = []
			bj = []
			for j in range(model_width):
				c = model.NewIntVar(0,1,'a{:d}{:d}{:d}'.format(p,i,j))
				d = model.NewBoolVar('b{:d}{:d}{:d}'.format(p,i,j))
				model.Add(c == 1).OnlyEnforceIf(d)
				model.Add(c == 0).OnlyEnforceIf(d.Not())
				aj.append(c)
				bj.append(d)
			ai.insert(i,aj)
			bi.insert(i,bj)
		a.insert(p,ai)
		b.insert(p,bi)

	## ADD LIFE TRANSITION RULES, INCLUDING PERIODICITY ##
	for p in range(period):
		for i in range(model_height):
			for j in range(model_width):
				# Cells in the ship channel are not forced to return to the original values
				if p == period-1 and (i,j) in ship_channel:
					continue
				ns = sum(a[p][i+k][j+m] for k in range(-1,2) for m in range(-1,2) if i+k in range(model_height) and j+m in range(model_width) and (k,m) != (0,0))
				model.Add(ns >= 2).OnlyEnforceIf([b[(p+1) % period][i][j],b[p][i][j]])
				model.Add(ns <= 3).OnlyEnforceIf([b[(p+1) % period][i][j],b[p][i][j]])
				model.Add(ns != 2).OnlyEnforceIf([b[(p+1) % period][i][j].Not(),b[p][i][j]])
				model.Add(ns != 3).OnlyEnforceIf([b[(p+1) % period][i][j].Not(),b[p][i][j]])
				model.Add(ns == 3).OnlyEnforceIf([b[(p+1) % period][i][j],b[p][i][j].Not()])
				model.Add(ns != 3).OnlyEnforceIf([b[(p+1) % period][i][j].Not(),b[p][i][j].Not()])

	## SET INITIAL AND BOUNDARY CONDITIONS IN PATTERN BOUNDING BOX ##
	
	# Now, let's set up the values we're forcing from the input pattern
	# For the rotor, we want to enforce the exact periodicity from the original pattern
	rotor_cells = set()
	for x in rotor:
		# We need to check if x is in the search space (think of removing rows). If not, we can throw an error here.
		if x[0] not in range(vert_search_offset,vert_search_offset+search_height) or x[1] not in range(horz_search_offset,horz_search_offset+search_width):
			sys.exit('INFEASIBLE: Rotor cell lies outside specified search space.')
		for p in range(period):
			model.Add(a[p][x[0]][x[1]] == x[2][p])
		rotor_cells.add((x[0],x[1]))	
			
	# For the remaining cells in the original pattern, we don't know what value they'll need, but we want
	# to force them to not join the rotor. Note: if preserved stator cells end up falling outside the search space, then
	# the model will (and should) return INFEASIBLE.
	for x in stator:
		if x[0] not in range(vert_search_offset,vert_search_offset+search_height) or x[1] not in range(horz_search_offset,horz_search_offset+search_width):
			if x in preserve:
				sys.exit('INFEASIBLE: Preserved stator cell lies outside specified search space.')
			# If x is NOT in preserve, then this is de-facto setting this cell to 0, which is OK.
		else:
			for p in range(1,period):
				model.Add(a[p][x[0]][x[1]] == a[0][x[0]][x[1]])
			if x in preserve:
				model.Add(a[0][x[0]][x[1]] == 1)
			if x in forceblank:
				model.Add(a[0][x[0]][x[1]] == 0)
				
	for x in blank:
		# Perfectly OK for x to be blank outside the search space, even if it is to be preserved.
		if x[0] in range(vert_search_offset,vert_search_offset+search_height) and x[1] in range(horz_search_offset,horz_search_offset+search_width):
			for p in range(1,period):
				model.Add(a[p][x[0]][x[1]] == a[0][x[0]][x[1]])
			if x in preserve:
				model.Add(a[0][x[0]][x[1]] == 0)
			
	## ADJUSTING INITIAL AND BOUNDARY CONDITIONS ##
	# Cells are outside the original pattern but in the search space will be static, unless they are in the ship channel
	for i in range(vert_search_offset,vert_search_offset+search_height):
		for j in range(horz_search_offset,horz_search_offset+search_width):
			if (i not in range(vert_pattern_offset,vert_pattern_offset+pattern_height) or j not in range(horz_pattern_offset,horz_pattern_offset+pattern_width)) and (i,j) not in ship_channel:
					for p in range(1,period):
						model.Add(a[p][i][j] == a[0][i][j])
	
	# All cells outside the search space need to be forced to 0, unless they are in the ship channel
	for i in range(vert_life_offset,vert_life_offset+life_height):
		for j in range(horz_life_offset,horz_life_offset+life_width):
			if (i not in range(vert_search_offset,vert_search_offset+search_height) or j not in range(horz_search_offset,horz_search_offset+search_width)) and (i,j) not in ship_channel:
				for p in range(period):
					model.Add(a[p][i][j] == 0)
		
	# Finally the outer one cell frame of the model. Needs to be zero in every period, unless it's in the ship channel
	for i in range(model_height):
		for j in range(model_width):
			if (i not in range(vert_life_offset,vert_life_offset+life_height) or j not in range(horz_life_offset,horz_life_offset+life_width)) and (i,j) not in ship_channel:
				for p in range(period):
					model.Add(a[p][i][j] == 0)
					
	##MODEL-TEST: for debugging##
	if model_test:
		live_cells,orig_height,orig_width = RLE_to_tuples(pattern)
		for i in range(orig_height):
			for j in range(orig_width):
				if (i,j) in live_cells:
					model.Add(a[0][i+vert_orig_offset][j+horz_orig_offset] == 1)
				else:
					model.Add(a[0][i+vert_orig_offset][j+horz_orig_offset] == 0)
					
	# Calculate size of stator
	size = sum(a[0][i][j] for i in range(vert_search_offset,vert_search_offset+search_height) for j in range(horz_search_offset,horz_search_offset+search_width) if (i,j) not in (rotor_cells | ship_channel))
	
	model.Minimize(size)

	solver = cp_model.CpSolver()
	print_vars = []
	for i in range(vert_search_offset,search_height+vert_search_offset):
		print_vars.insert(i-vert_search_offset,[a[0][i][j] for j in range(horz_search_offset,search_width+horz_search_offset)])
	solution_callback = ObjectiveSolutionPrinter(print_vars)
	status = solver.Solve(model)
	P.print(P.TEST,'Status = %s' % solver.StatusName(status))
	value_vars = [[solver.Value(print_vars[i][j]) for j in range(search_width)] for i in range(search_height)]
	rle = array_to_RLE(value_vars)
	P.print(P.NORMAL,rle)
	P.print(P.TEST,'Size of new stator: '+ str(solver.Value(size)))
		#for p in range(period):
		#	for i in range(model_height):
		#		print('{:d}:'.format(i),end='')
		#		for j in range(model_width):
		#			print('{:d}'.format(solver.Value(a[p][i][j])),end=" ")
		#		print()
		#	print()

if __name__ == '__main__':
	# Set up argument parsing
	parser = argparse.ArgumentParser(prog='stator_reducer',description='A program to minimize the stator size of oscillators and guns in Conway\'s Game of Life',epilog='The primary difference between gun processing and oscillator processing is that for oscillator processing, any cells in the pattern evolution that leave the original bounding box are assumed to be part of the oscillator, and the original bounding box is expanded. For gun processing, any cells in the pattern that leave the original bounding box are assumed to be part of the ship channel.')
	parser.add_argument('pattern', help='Filename of RLE containing object to be reduced')
	parser.add_argument('period',help='Period of the object')
	parser.add_argument('--adjust',help='Adjusts the size of the target bounding box to search. Must be followed by four integers, indicating the change on the left, right, top, and bottom of the bounding box. Negative numbers decrease the size of the bounding box, positives increase it.',nargs=4,default=['0','0','0','0'],metavar=('LEFT','RIGHT','TOP','BOTTOM'))
	parser.add_argument('--preserve',help='Filename of RLE where all cells containing in 1 will be preserved in the reduced object. Should be the same size as the pattern. Useful if you\'re trying to preserve clearance around a sparker.',default=None)
	parser.add_argument('--forceblank',help='Filename of RLE where all cells containing the 1 mask stator cells in the same original pattern, forcing them to be zero in the modified oscillator. Useful for creating sparker variants with additional clearance.',default=None)
	parser.add_argument('--gun',help='Processes the pattern as if it\'s a gun.',action='store_true')
	parser.add_argument('--shipchannel',help='Filename of RLE which identifies all cells used by spaceships created by the gun within the original bounding box. Should be the same size as the pattern.',default=None)
	parser.add_argument('--verbose',default=1,help='Set level of print verbosity: 0 is TEST (least), 2 is DEBUG (most). Defaults to 1.')
	parser.add_argument('--modeltest',help='Tests model against input pattern, which should always yield a solution unless grid is being shrunk.',action='store_true')
		
	args = parser.parse_args()

	# Open files
	with open(args.pattern, 'r') as pattern_file:
		try:
			pattern = pattern_file.read()
		except:
			print('Could not open file' + args.pattern)
	
	if args.preserve is not None:
		with open(args.preserve, 'r') as preserve_file:
			try:
				preserveRLE = preserve_file.read()
			except:
				print('Could not open file' + args.preserve)
		preserve,height,width = RLE_to_tuples(preserveRLE)
	else:
		preserve = set()
				
	if args.forceblank is not None:
		with open(args.forceblank, 'r') as forceblank_file:
			try:
				forceblankRLE = forceblank_file.read()
			except:
				print('Could not open file' + args.forceblank)
		forceblank,height,width = RLE_to_tuples(forceblankRLE)
	else:
		forceblank = set()

	if args.shipchannel is not None:
		with open(args.shipchannel, 'r') as shipchannel_file:
			try:
				shipchannelRLE = shipchannel_file.read()
			except:
				print('Could not open file' + args.ship-channel)
		shipchannel,height,width = RLE_to_tuples(shipchannelRLE)
	else:
		shipchannel = set()
		
	adjust = list(map(int,args.adjust))

	# Set up screen printer
	P = screen_printer(int(args.verbose))
	
	main(pattern,int(args.period),adjust[0],adjust[1],adjust[2],adjust[3],preserve,forceblank,args.gun,shipchannel,args.modeltest)
