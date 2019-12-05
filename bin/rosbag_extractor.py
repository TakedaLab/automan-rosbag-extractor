#!/usr/bin/env python
import argparse
import json
import traceback
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
from rosbag.bag import Bag
import sys
from pypcd import PointCloud
sys.path.append(os.path.join(os.path.dirname(__file__), '../libs'))
from core.storage_client_factory import StorageClientFactory
from core.automan_client import AutomanClient


class UnknownCalibrationFormatError(Exception):
    pass


class RosbagExtractor(object):

    @classmethod
    def extract(cls, automan_info, file_path, output_dir, raw_data_info, calibfile=None):
        extrinsics_mat, camera_mat, dist_coeff = None, None, None
        if calibfile:
            try:
                calib_path = calibfile
                extrinsics_mat, camera_mat, dist_coeff = cls.__parse_calib(calib_path)
            except Exception:
                raise UnknownCalibrationFormatError
        available_candidates, available_topics = cls.__get_candidates(
            automan_info, int(raw_data_info['project_id']), int(raw_data_info['original_id']))

        candidates = filter(lambda c: c['candidate_id'] in raw_data_info['candidates'], available_candidates)
        topics_to_extract = [c['topic_name'] for c in candidates]
        topic_msgs = {topic_name: None for topic_name in topics_to_extract}

        try:
            count = 0
            with Bag(file_path) as bag:
                for topic, msg, t in bag.read_messages():
                    if topic in topics_to_extract:
                        topic_msgs[topic] = msg
                    if all([m is not None for m in topic_msgs.values()]) and len(topic_msgs) > 0:
                        count += 1
                        for c in candidates:
                            save_msg = topic_msgs[c['topic_name']]
                            output_path = output_dir + str(c['candidate_id']) \
                                + '_' + str(count).zfill(6)
                            if 'PointCloud2' in c['msg_type']:
                                cls.__process_pcd(save_msg, output_path)
                            elif 'Image' in c['msg_type']:
                                cls.__process_image(
                                    save_msg, type(save_msg).__name__, output_path, camera_mat, dist_coeff)
                            else:
                                raise NotImplementedError('Unsupported message type: {}'.format(c['msg_type']))
                        topic_msgs = {topic_name: None for topic_name in topics_to_extract}

            result = {
                'file_path': output_dir,
                'frame_count': count,
                'name': os.path.basename(path),  # FIXME
                'original_id': int(raw_data_info['original_id']),
                'candidates': raw_data_info['candidates'],
            }
            return result
        except Exception as e:
            # FIXME
            print(traceback.format_exc())
            raise(e)

    @staticmethod
    def __get_candidates(automan_info, project_id, original_id):
        path = '/projects/' + str(project_id) + '/originals/' + str(original_id) + '/candidates/'
        res = AutomanClient.send_get(automan_info, path).json()
        candidates = []
        topics = []
        for c in res["records"]:
            analyzed_info = json.loads(c['analyzed_info'])
            candidate = {
                'candidate_id': c["candidate_id"],
                'msg_type': analyzed_info['msg_type'],
                'topic_name': analyzed_info['topic_name']
            }
            candidates.append(candidate)
            topics.append(analyzed_info['topic_name'])
        return candidates, topics

    @staticmethod
    def __process_pcd(msg, file_path):
        pc = PointCloud.from_msg(msg)
        pc.save(file_path + '.pcd')

    @staticmethod
    def __process_image(msg, _type, file_path, camera_mat=None, dist_coeff=None):
        if "Compressed" in _type:
            bridge = CvBridge()
            image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        else:
            bridge = CvBridge()
            image = bridge.imgmsg_to_cv2(msg, "bgr8").astype('f')

        if camera_mat and dist_coeff:
            image = cv2.undistort(image, camera_mat, dist_coeff, None, camera_mat)

        cv2.imwrite(file_path + ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    @staticmethod
    def __parse_calib(calib_path):
        fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
        camera_extrinsic_mat = fs.getNode("CameraExtrinsicMat").mat()
        camera_mat = fs.getNode("CameraMat").mat()
        dist_coeff = np.transpose(fs.getNode("DistCoeff").mat())
        return camera_extrinsic_mat, camera_mat, dist_coeff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_type', required=True)
    parser.add_argument('--storage_info', required=True)
    parser.add_argument('--automan_info', required=True)
    parser.add_argument('--raw_data_info', required=True)
    args = parser.parse_args()

    storage_client = StorageClientFactory.create(
        args.storage_type,
        json.loads(args.storage_info)
    )
    storage_client.download()
    path = storage_client.get_input_path()
    output_dir = storage_client.get_output_dir()
    os.makedirs(output_dir)
    res = RosbagExtractor.extract(
        json.loads(args.automan_info), path, output_dir, json.loads(args.raw_data_info))
    AutomanClient.send_result(json.loads(args.automan_info), res)
