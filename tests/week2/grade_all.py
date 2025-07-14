from graderlib import grader_summary

from grade_part1 import testsets_part1
from grade_part2 import testsets_part2
from grade_part3 import testsets_part3
from grade_part4 import testsets_part4
from grade_part5 import testsets_part5
from grade_part6 import testsets_part6
from grade_part7 import testsets_part7

def testsets_all():
  testsets_part1()
  testsets_part2()
  testsets_part3()
  testsets_part4()
  testsets_part5()
  testsets_part6()
  testsets_part7()
  
if __name__ == "__main__":
  testsets_all()
  
  grader_summary("all")