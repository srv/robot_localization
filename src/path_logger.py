#!/usr/bin/env python
import rospy
from robot_localization.srv import GetPath, GetPoseRequest, GetPoseResponse, GetPose
from geometry_msgs.msg import PoseWithCovarianceStamped
import csv

class PathLogger:
  def __init__(self):
    rospy.init_node('path_logger')
    rospy.loginfo('[PathLogger] Running!')
    self.path = []
    if rospy.has_param('/path_logger/csv_filename'):
      self.csv_filename = rospy.get_param('/path_logger/csv_filename')
      rospy.loginfo('[PathLogger] Reading CSV...')
      self.readPath()
      # Provide a service to get the pose from a particular stamp
      self.server = rospy.Service('/path_logger/get_state', GetPose, self.getStateSrvCallback)
      rospy.spin()
    else:
      rospy.loginfo('[PathLogger] Trying to get path from service')
      self.getPath()
      self.savePath()
      # Print info about how long is the path and where it is


  def readPath(self):
    reader = csv.DictReader(self.csv_filename)
    header = True
    for s in csv.reader(open(self.csv_filename, "rb")):
      if header:
        header = False
        continue
      p = PoseWithCovarianceStamped()
      p.header.frame_id = 'map'
      p.header.stamp = rospy.Time.from_sec(float(s[0]))
      p.header.seq = len(self.path)
      p.pose.pose.position.x = float(s[1])
      p.pose.pose.position.y = float(s[2])
      p.pose.pose.position.z = float(s[3])
      p.pose.pose.orientation.x = float(s[4])
      p.pose.pose.orientation.y = float(s[5])
      p.pose.pose.orientation.z = float(s[6])
      p.pose.pose.orientation.w = float(s[7])
      self.path.append(p)
    print('Added ' + str(len(self.path)) + ' poses to the path')

  def savePath(self):
    try:
      with open('states.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['stamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        for data in self.path:
          stamp = data.header.stamp.to_sec()
          x = data.pose.pose.position.x
          y = data.pose.pose.position.y
          z = data.pose.pose.position.z
          qx = data.pose.pose.orientation.x
          qy = data.pose.pose.orientation.y
          qz = data.pose.pose.orientation.z
          qw = data.pose.pose.orientation.w
          writer.writerow([stamp, x, y, z, qx, qy, qz, qw])
        csvfile.close()
        print 'Successfully saved ' + str(len(self.path)) + ' poses in states.csv'
    except IOError as (errno, strerror):
      print("I/O error({0}): {1}".format(errno, strerror))

  def getStateSrvCallback(self, request):
    stamp = rospy.Time.from_sec(float(request.stamp))
    last_p = []
    for p in self.path:
      if last_p:
        if p.header.stamp - stamp > rospy.Duration(0) and stamp - last_p.header.stamp > rospy.Duration(0):
          interp_p = self.interpolate(last_p, p, stamp)
          # print 'Requested pose in ' + str(last_p.header.seq)
          return GetPoseResponse(interp_p)
      last_p = p
    print 'Requested time: ' + str(stamp.to_sec()) + ' available poses from: ' + str(self.path[0].header.stamp.to_sec()) + ' to ' + str(self.path[-1].header.stamp.to_sec())

  def getPath(self):
    rospy.wait_for_service('/ekf_map/get_path')
    try:
      get_path = rospy.ServiceProxy('/ekf_map/get_path', GetPath)
      response = get_path(smoothed = True)
      self.path = response.poses
    except rospy.ServiceException, e:
      print "Service call failed: %s"%e

  def interpolate(self, last_p, p, stamp):
    interp_p = last_p
    s1 = last_p.header.stamp
    s2 = p.header.stamp
    interp_p.pose.pose.position.x = self.interp(stamp, s1, last_p.pose.pose.position.x, s2, p.pose.pose.position.x)
    interp_p.pose.pose.position.y = self.interp(stamp, s1, last_p.pose.pose.position.y, s2, p.pose.pose.position.y)
    interp_p.pose.pose.position.z = self.interp(stamp, s1, last_p.pose.pose.position.z, s2, p.pose.pose.position.z)
    interp_p.pose.pose.orientation.x = self.interp(stamp, s1, last_p.pose.pose.orientation.x, s2, p.pose.pose.orientation.x)
    interp_p.pose.pose.orientation.y = self.interp(stamp, s1, last_p.pose.pose.orientation.y, s2, p.pose.pose.orientation.y)
    interp_p.pose.pose.orientation.z = self.interp(stamp, s1, last_p.pose.pose.orientation.z, s2, p.pose.pose.orientation.z)
    interp_p.pose.pose.orientation.w = self.interp(stamp, s1, last_p.pose.pose.orientation.w, s2, p.pose.pose.orientation.w)
    interp_p.header.stamp = stamp
    return interp_p

  def interp(self, t, t0, x0, t1, x1):
    return x0 + (x1-x0)/(t1.to_sec()-t0.to_sec())*(t.to_sec()-t0.to_sec())



if __name__ == '__main__':
  try:
    PathLogger()
  except rospy.ROSInterruptException:
    pass