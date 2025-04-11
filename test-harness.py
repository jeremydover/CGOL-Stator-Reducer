import unittest
import subprocess
import os
import shutil
import re
  
class TestCases(unittest.TestCase):
	
	def out_test(self,insize,status,outsize):
		return 'Sizeofinputstator:'+str(insize)+'Status='+status+'Sizeofnewstator:'+str(outsize)
	
	def test_123_same_box(self):
		result = subprocess.run([shutil.which('python'),os.path.join(os.getcwd(),'stator_reducer.py'),os.path.join(os.getcwd(),'tests','123.rle'),'3','--verbose','0'], capture_output=True, text=True)
		test_string = re.sub("\s+", "", result.stdout)
		self.assertEqual(test_string,self.out_test(25,'OPTIMAL',25),'Failed 123 same box')
	
	def test_123_wider_box(self):
		result = subprocess.run([shutil.which('python'),os.path.join(os.getcwd(),'stator_reducer.py'),os.path.join(os.getcwd(),'tests','123.rle'),'3','--verbose','0','--adjust','5','5','0','0'], capture_output=True, text=True)
		test_string = re.sub("\s+", "", result.stdout)
		self.assertEqual(test_string,self.out_test(25,'OPTIMAL',25),'Failed 123.rle wider box')
		
	def test_123_higher_box(self):
		result = subprocess.run([shutil.which('python'),os.path.join(os.getcwd(),'stator_reducer.py'),os.path.join(os.getcwd(),'tests','123.rle'),'3','--verbose','0','--adjust','0','0','5','5'], capture_output=True, text=True)
		test_string = re.sub("\s+", "", result.stdout)
		self.assertEqual(test_string,self.out_test(25,'OPTIMAL',23),'Failed 123.rle higher box')
		
	def test_246p3(self):
		result = subprocess.run([shutil.which('python'),os.path.join(os.getcwd(),'stator_reducer.py'),os.path.join(os.getcwd(),'tests','246p3-karel-domino.rle'),'3','--verbose','0','--adjust','1','1','0','0'], capture_output=True, text=True)
		test_string = re.sub("\s+", "", result.stdout)
		self.assertEqual(test_string,self.out_test(158,'OPTIMAL',147),'Failed 246p3-karel-domino.rle')
		
	def test_p14_glider_gun(self):
		result = subprocess.run([shutil.which('python'),os.path.join(os.getcwd(),'stator_reducer.py'),os.path.join(os.getcwd(),'tests','p14-glider-gun.rle'),'14','--verbose','0','--gun'], capture_output=True, text=True)
		test_string = re.sub("\s+", "", result.stdout)
		self.assertEqual(test_string,self.out_test(491,'OPTIMAL',489),'Failed p14-glider-gun.rle')
		
	def test_great_on_off_OT(self):
		result = subprocess.run([shutil.which('python'),os.path.join(os.getcwd(),'stator_reducer.py'),os.path.join(os.getcwd(),'tests','great-on-off.rle'),'2','--verbose','0'], capture_output=True, text=True)
		test_string = re.sub("\s+", "", result.stdout)
		self.assertEqual(test_string,self.out_test(18,'OPTIMAL',6),'Failed great-on-off.rle')
		
	def test_p5_b36_s245(self):
		result = subprocess.run([shutil.which('python'),os.path.join(os.getcwd(),'stator_reducer.py'),os.path.join(os.getcwd(),'tests','p5-b36-s245.rle'),'5','--verbose','0'], capture_output=True, text=True)
		test_string = re.sub("\s+", "", result.stdout)
		self.assertEqual(test_string,self.out_test(469,'OPTIMAL',344),'Failed p5-b36-s245.rle')
		
if __name__ == '__main__':
    unittest.main()