# Pose Estimation for a Virtual Reality HMD and Controllers

- **Author**: Shawn Oppermann
- **Term**: Summer 2023

## Premise

Virtual Reality (VR) gaming is rising in popularity as a pastime, but not without issue. The standard VR setup consists of a head-mounted display (HMD) and two controllers. This alone may be hard to afford, but adding full-body tracking can be prohibitively expensive, as each tracked joint requires a separate tracker.

Popular VR games such as VRChat have a standard rig that includes joints for the hands, feet, elbows, knees, hips, chest, neck, shoulders, and head. Most users only have three trackers for the hands and head, meaning the rest of the joint positions are inferred. While inverse kinematics (IK) can make a fast, reasonable inference for individual limbs, the error between the true and inferred joint positions across the entire rig can often be quite large.

VR software supports additional trackers to reduce inference, but this approach can become cumbersome and unaffordable. To the end of avoiding additional hardware, the proposed project aims to explore the applications of various regression models (most likely with an emphasis on deep learning) for pose estimation, comparing the speed and accuracy of different approaches. In other words, the project should answer the question of whether it is feasible to predict the pose of an entire human body using only the location and orientation of a few joints, initially just the head and hands.

## Dataset

Carnegie Mellon University researchers have courteously provided an open source motion capture database. The data consists of the joint positions and orientations for various motion-captured actions. For example, there are entries for men and women running, playing sports, sitting, etc. for 30 or more frames. 

The joints for a standard VR rig are a subset of the joints used in the dataset.

```
└── root
    ├── lhipjoint
    │   └── lfemur
    │       └── ltibia
    │           └── lfoot
    │               └── ltoes
    ├── rhipjoint
    │   └── rfemur
    │       └── rtibia
    │           └── rfoot
    │               └── rtoes
    └── lowerback
        └── upperback
            └── thorax
                ├── lowerneck
                │   └── upperneck
                │       └── head
                ├── lclavicle
                │   └── lhumerus
                │       └── lradius
                │           └── lwrist
                │               ├── lthumb
                │               └── lhand
                │                   └── lfingers
                └── rclavicle
                    └── rhumerus
                        └── rradius
                            └── rwrist
                                ├── rthumb
                                └── rhand
                                    └── rfingers
```

## Process

This research aims to compare the speed and accuracy of inference using various deep learning techniques such as Convolutional Neural Networks (CNNs), Dense Neural Networks (DNNs), and Long-Short Term Memory Networks (LSTMs), and perhaps Transformers. The control approaches will consist of FABRIK and CCD.

The training process will involve masking various joints and then training each model on the dataset, predicting the location of the masked joints. Joints taken into consideration may vary depending on what is tracked by hardware. For example, it is hard to justify inferring the orientation of the legs if there is no information provided about them.

Runtime will be measured by running inference on a test set and recording the total time elapsed. Accuracy will refer to the mean-squared error (MSE) of all inferred joints when compared to the true joint positions. It is important to find a solver that not only returns accurate results, but also has the ability to run in realtime.
