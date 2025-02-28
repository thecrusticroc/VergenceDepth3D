import logging
import numpy as np
import csv
import os
from plugin import Plugin

CUSTOM_TOPIC = "Vergence_Data"
logger = logging.getLogger(__name__)

class Vergence(Plugin):
    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.order = 0.01
        self.vergence_data = []  # List for storage in event

    def recent_events(self, events):
        if "gaze" not in events or not events["gaze"]:
            return  # No Gaze data

        # Only most recent gaze
        recent_gaze_data = events["gaze"][-1]
        eye_centers_3d = recent_gaze_data.get("eye_centers_3d", [])
        gaze_normals_3d = recent_gaze_data.get("gaze_normals_3d", [])

        time = recent_gaze_data.get("timestamp", 0)
        theta_deg = -1
        angle_deg = -1

        if len(eye_centers_3d) == 2 and len(gaze_normals_3d) == 2:
            o1 = np.array(eye_centers_3d.get("0"))
            o2 = np.array(eye_centers_3d.get("1"))
            d1 = np.array(gaze_normals_3d.get("0"))
            d2 = np.array(gaze_normals_3d.get("1"))
            w = o1 - o2

            # Normalize
            d1 /= np.linalg.norm(d1)
            d2 /= np.linalg.norm(d2)

            # Dot products
            d1_dot_d2 = np.dot(d1, d2)
            d1_dot_w = np.dot(d1, w)
            d2_dot_w = np.dot(d2, w)

            # Solve linear equation
            denom = 1 - d1_dot_d2 * d1_dot_d2
            if np.abs(denom) > 1e-6:
                t = (d1_dot_d2 * d2_dot_w - d1_dot_w) / denom
                s = (d2_dot_w - d1_dot_d2 * d1_dot_w) / denom

                p1 = o1 + t * d1
                p2 = o2 + s * d2
                pm = (p1 + p2) / 2

                # IPD + Distance method
                ipd = np.linalg.norm(o1 - o2)
                om = (o1 + o2) / 2
                distance = np.linalg.norm(om - pm)
                theta_rad = 2 * np.arctan(ipd / (2 * distance))
                theta_deg = np.degrees(theta_rad)

                # Gaze normals method
                norm_d1 = np.linalg.norm(d1)
                norm_d2 = np.linalg.norm(d2)
                cos_theta = d1_dot_d2 / (norm_d1 * norm_d2)
                angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)

                timestamp = self.g_pool.get_timestamp()
                self.vergence_data.append([timestamp, theta_deg, angle_deg])

                # Enable this for debug angle information during Pupil Labs Capture
                #logger.info(f"t: {time}, Vergence Angle (IPD): {theta_deg}, Vergence Angle (Dot): {angle_deg}")

        # save event for pupil capture
        custom_datum = {
            "topic": CUSTOM_TOPIC,
            "timestamp": time,
            "vergence IPD": theta_deg,
            "vergence Dot": angle_deg
        }
        events[CUSTOM_TOPIC] = [custom_datum]


"""
Attempt to export vergence events as csv. Ignore this for now
    def on_stop(self):
        # Save data to csv after runtime
        if not self.vergence_data:
            logger.info(f"no Data")
            return

        recording_path = self.g_pool.rec_dir
        csv_path = os.path.join(recording_path, "vergence_data.csv")

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "vergence IPD", "vergence Dot"])
            writer.writerows(self.vergence_data)

        logger.info(f"CSV Done")
        print(f"Vergence-data saved at: {csv_path}")
"""