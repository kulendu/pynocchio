import signal

import pyglet
from pyglet.gl import *
from pynocchio import auto_rig, Mesh, Vector3, Transform, VectorTransform, Points
from pynocchio import skeletons

# SMPL imports
from smplx import SMPL
import numpy as np
import torch
import pickle

class Animation:
    def __init__(self, transforms_count, axis, stretch, delta):
        self._axis = axis
        self._stretch = stretch
        self._delta = delta
        self._delta_sign = 1.
        self._transform_index = 0

        self._translates = [[0., 0., 0.] for i in range(transforms_count)]

    def step(self):
        value = self._translates[self._transform_index][self._axis]
        value = value + self._delta_sign * self._delta
        if value >= self._stretch:
            value = self._stretch
            self._delta_sign = self._delta_sign * -1
        if value <= 0.:
            value = 0.
            if self._delta_sign < 0:
                self._delta_sign = self._delta_sign * -1
                self._transform_index = 0 \
                    if self._transform_index + 1 == len(self._translates) else self._transform_index + 1
        self._translates[self._transform_index][self._axis] = value
        return VectorTransform([Transform(Vector3(t[0], t[1], t[2])) for t in self._translates])


class Model:
    def __init__(self, mesh):
        self._batch = pyglet.graphics.Batch()
        vertices = [edge.vertex for edge in mesh.edges]
        positions = Model.get_positions(mesh)
        self._vertex_list = self._batch.add_indexed(len(mesh.vertices), GL_TRIANGLES, None,
                                                    vertices, ('v3d', tuple(positions)))

    @staticmethod
    def get_positions(mesh):
        positions = []
        for vertex in mesh.vertices:
            position = vertex.position
            positions.append(position[0])
            positions.append(position[1])
            positions.append(position[2])
        return positions

    def update(self, mesh):
        self._vertex_list.vertices = Model.get_positions(mesh)

    def draw(self):
        self._batch.draw()


class Bones:
    def __init__(self, skeleton, embedding):
        self._batch = pyglet.graphics.Batch()
        positions = Bones.get_positions(skeleton, embedding)
        self._vertex_list = self._batch.add((len(skeleton.parent_indices) - 1) * 2, GL_LINES, None, ('v3d', positions))

    @staticmethod
    def get_positions(skeleton, embedding):
        positions = []
        for i in range(1, len(skeleton.parent_indices)):
            j = skeleton.parent_indices[i]
            v1 = embedding[i]
            v2 = embedding[j]
            positions.append(v1[0])
            positions.append(v1[1])
            positions.append(v1[2])
            positions.append(v2[0])
            positions.append(v2[1])
            positions.append(v2[2])
        return positions

    def update(self, skeleton, embedding):
        self._vertex_list.vertices = Bones.get_positions(skeleton, embedding)

    def draw(self):
        self._batch.draw()


# human_mesh = Mesh('data/sveta.obj')
# human_skeleton = skeletons.HumanSkeleton()
# human_skeleton.scale(0.7)
# attach = auto_rig(human_skeleton, human_mesh)

# model = Model(human_mesh)
# bones = Bones(human_skeleton, attach.embedding)
# window = pyglet.window.Window()
# animation = Animation(len(attach.embedding), 0, 0.5, 0.05)



# Define the SMPL model and get the joint locations 
import smplx
smpl_path = '/home/kulendu/rabit_kulendu/pynocchio/data/SMPL_NEUTRAL.pkl'
smpl = smplx.create(model_path=smpl_path, model_type='smpl', gender='neutral', create_transl=False)

betas = torch.zeros(1, 10) 
body_pose = torch.zeros(1, 69)  
global_orient = torch.zeros(1, 3)  

output = smpl(betas=betas, body_pose=body_pose, global_orient=global_orient, return_verts=True)


# getting the parent joint
with open(smpl_path, 'rb') as f:
    smpl_model_data = pickle.load(f, encoding='latin1')

kintree_table = smpl_model_data['kintree_table']
parent_joints = kintree_table[0] 

for idx, parent_joint in enumerate(parent_joints):
    print(f"{idx} {parent_joint}")

joints = output.joints.squeeze(0)[:24] # First 24 joints for the SMPL model
breakpoint()

# dumping the joint info in a file
def joint_dump(filename):
    print(f"----- Saving the joint info in {filename} ---------")
    with open(filename, 'w') as file:
        for parent, (idx, joint) in zip(parent_joints, enumerate(joints)):
            line = f"{idx} {joint[0]} {joint[1]} {joint[2]} {parent}\n"
            file.write(line)

    print(f"----- Saved Successfully! ---------")

# joint_dump('smpl_skeleton_new.out')


# Define Mesh and Skeleton for RaBit
rabit_mesh = Mesh('data/rabit.obj')
rabit_skeleton = skeletons.FileSkeleton('smpl_skeleton_new.out')
rabit_skeleton.scale(0.7)
attach = auto_rig(rabit_skeleton, rabit_mesh)

model = Model(rabit_mesh)
bones = Bones(rabit_skeleton, attach.embedding)
window = pyglet.window.Window()
animation = Animation(len(attach.embedding), 0, 0.5, 0.05)



def setup():
    glTranslatef(window.width // 2, window.height // 2, 0.)
    glScalef(200., 200., 1.)


def update(dt):
    transforms = animation.step()

    mesh_transformed = attach.deform(rabit_mesh, transforms)
    model.update(mesh_transformed)

    embedding_transformed = Points()
    embedding_transformed.append(attach.embedding[0])
    for i in range(1, len(transforms)):
        embedding_transformed.append(transforms[i - 1] * attach.embedding[i])
    bones.update(rabit_skeleton, embedding_transformed)


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
    glColor3f(1., 1., 1.)
    model.draw()
    glColor3f(0., .5, 0.)
    bones.draw()


pyglet.clock.schedule_interval(update, 1. / 60)
setup()
pyglet.app.run()
