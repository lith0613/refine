
import os, sys
import subprocess
import numpy as np
import scipy.io as sio
from skimage import io
import cv2
import numpy as np
import os

import torch
import matplotlib.pyplot as plt
import trimesh
import igl
import time

def load_obj(file):
    bs_dict = {}
    f = open(file, "r")
    lines = f.readlines()

    #verts
    verts = [list(map(float, s.strip().split()[1:4])) for s in lines if s.startswith("v ")]
    verts = np.array(verts)
    bs_dict['verts'] = verts
    # print(verts[:3,:])

    #triangles
    triangles = [list(s.strip().split()[1:]) for s in lines if s.startswith("f ")]
    # print(triangles[:3])

    triangles_list_v = []
    for items in triangles:
        tri_list_v = []
        for item in items:
            tri_list_v.append(int(item.split('/')[0]))
        triangles_list_v.append(tri_list_v)

    triangles_v = np.array(triangles_list_v)
    bs_dict['triangles_v'] = triangles_v - 1    #face3d里面好像是从0开始索引

    return bs_dict


start = time.time()


#step 1 : load mesh
bs_dict = load_obj('1_0261.obj')
tris = bs_dict['triangles_v']
verts = bs_dict['verts']


#step 2 :subdivide
def subdiv(verts, tris, texcoords=None, face_index=None):
    if face_index is None:
        face_index = np.arange(len(tris))
    else:
        face_index = np.asanyarray(face_index)

    # the (c, 3) int array of vertex indices
    tris_subset = tris[face_index]

    # find the unique edges of our faces subset
    edges = np.sort(trimesh.remesh.faces_to_edges(tris_subset), axis=1)
    unique, inverse = trimesh.remesh.grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = verts[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(verts)

    # the new faces_subset with correct winding
    f = np.column_stack([tris_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         tris_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         tris_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces_subset per old face
    new_faces = np.vstack((tris, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]
    new_vertices = np.vstack((verts, mid))

    if texcoords is not None:
        texcoords_mid = texcoords[edges[unique]].mean(axis=1)
        new_texcoords = np.vstack((texcoords, texcoords_mid))
        return new_vertices, new_faces, new_texcoords

    return new_vertices, new_faces


for _ in range(1):
    verts, tris = subdiv(verts, tris)


#step 3 : check projection
def _rotate(points, rot_vec):

    theta = np.linalg.norm(rot_vec)
    with np.errstate(invalid='ignore'):
        v = rot_vec / theta
        v = np.nan_to_num(v)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + (points.dot(v.T) * (1 - cos_theta)).dot(v)

def project(points, rot_vec, scale, trans, keepz=False):
    points_proj = _rotate(points, rot_vec.reshape(1, 3))
    points_proj = points_proj * scale
    if keepz:
        points_proj[:, 0:2] = points_proj[:, 0:2] + trans
    else:
        points_proj = points_proj[:, 0:2] + trans
    return points_proj

#project according to the optimized params.
rot_vec = np.array([-0.1223821 , -0.04296856,  0.0422828 ])
scale = 4.447246635228211
trans = np.array([471.23514288, 767.65440025])
bs_vertices = project(verts, rot_vec, scale, trans, keepz=True)

#
img_1 = cv2.imread('1_0261.jpg')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)/255.0
h,w,c = img_1.shape
bs_vertices[:,1] = h - bs_vertices[:,1]
for u,v in bs_vertices[:,:2]:
    if 0<u < w and 0<v<h:
        img_1[int(v), int(u), :] = (0.0, 1.0, 0.0)
plt.figure(figsize=(10, 10))
plt.imshow(img_1)
plt.show()



#step 4 : backproject the image to the mesh surface (pytorch grid sample)
device = torch.device('cpu')
img_11 = cv2.imread('1_0261.jpg')
img_3 = cv2.cvtColor(img_11, cv2.COLOR_RGB2GRAY)
# blurred = cv2.GaussianBlur(img_3, (3,3), 0);
# img_3 = cv2.Laplacian(img_3, cv2.CV_16S, ksize=3)
# img_3 = cv2.convertScaleAbs(img_3)
# plt.imshow((img_3)/255.0)
# plt.show()

image_intensity = torch.from_numpy(img_3[:,:,np.newaxis]/255.0).float().permute(2, 0, 1).unsqueeze(0)  # 1*C*w*h
geometry_xy = torch.from_numpy(bs_vertices[:,:2]).float()
geometry_xy = geometry_xy.view(1, 1, -1, 2)
# change range to [-1,1]
geometry_xy[:, :, :, 0] = (geometry_xy[:, :, :, 0] / (w - 1)) * 2 - 1
geometry_xy[:, :, :, 1] = (geometry_xy[:, :, :, 1] / (h - 1)) * 2 - 1

geometry_intensity = torch.nn.functional.grid_sample(image_intensity, geometry_xy, align_corners=True)
geometry_intensity = geometry_intensity.permute(0, 2, 3, 1).view(-1, 1)

# geometry_intensity[bs_vertices[:,2]<0] = torch.FloatTensor([1.0])



#step 5 : filtering the high frequency part using cotangent weight discrete Laplacian operator
def laplacian_cot(verts_packed, faces_packed):
    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.

    Args:
        meshes: Meshes object with a batch of meshes.
    Returns:
        2-element tuple containing
        - **L**: FloatTensor of shape (V,V) for the Laplacian matrix (V = sum(V_n))
           Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
           See the description above for more clarity.
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
           face areas containing each vertex
    """
    # V = sum(V_n), F = sum(F_n)
    V, F = verts_packed.shape[0], faces_packed.shape[0]

    face_verts = verts_packed[faces_packed]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces_packed[:, [1, 2, 0]]
    jj = faces_packed[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # For each vertex, compute the sum of areas for triangles containing it.
    idx = faces_packed.view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts_packed.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas


verts_packed = torch.from_numpy(verts).float()
faces_packed = torch.from_numpy(tris).long()
L, inv_areas = laplacian_cot(verts_packed, faces_packed)
vers_n, _ = geometry_intensity.shape


t1 = time.time()

#get the vertice tex though laplacian value( using our method)
# norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
# idx = norm_w > 0
# norm_w[idx] = 1.0 / norm_w[idx]
# tex_gray = geometry_intensity - L.mm(geometry_intensity) * norm_w

#according to matan's paper
dt = 0.2
ind_i = torch.Tensor(list(range(vers_n))).long().view(1,-1)
ind_j = torch.Tensor(list(range(vers_n))).long().view(1,-1)
indices = torch.cat((ind_i, ind_j), dim=0)
values = torch.ones(vers_n)
I = torch.sparse.FloatTensor(indices, values, (vers_n, vers_n))
tex_gray = geometry_intensity - (I - dt*L).to_dense().inverse().mm(geometry_intensity)

#
# tex_show = tex_gray.detach().cpu().numpy()
# for i,(u,v) in enumerate(bs_vertices[:,:2]):
#     if 0<u < w and 0<v<h:
#         img_1[int(v), int(u), :] = (tex_show[i], 0, 0)
# plt.figure(figsize=(10, 10))
# plt.imshow(img_1)
# plt.show()
# print(time.time()-t1)

#step 6 : get one-ring neighbor
one_ring = {}
for i,j, k in tris:
    if i not in one_ring.keys():
        one_ring[i] = [j, k]
    else:
        one_ring[i].extend([j, k])

    if j not in one_ring.keys():
        one_ring[j] = [i, k]
    else:
        one_ring[j].extend([i, k])

    if k not in one_ring.keys():
        one_ring[k] = [i, j]
    else:
        one_ring[k].extend([i, j])

for key, value in one_ring.items():
    one_ring[key] = set(value)

one_ring_sort = sorted(one_ring.items(), key= lambda i:i[0])


#step 7 : calculate grad photo loss
def get_verts_normals(verts_packed, faces_packed):

    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    # NOTE: this is already applying the area weighting as the magnitude
    # of the cross product is 2 x area of the triangle.
    # pyre-fixme[16]: `Tensor` has no attribute `index_add`.
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 1],
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 2],
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 0],
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
            dim=1,
        ),
    )

    verts_normals_packed = torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )

    return verts_normals_packed

verts_normals = get_verts_normals(verts_packed, faces_packed)
delta_u = torch.zeros_like(tex_gray, dtype=torch.float32)

for i in range(len(delta_u)):
    vertice = verts_packed[i]
    neighbors = one_ring_sort[i][1]

    fenmu = torch.FloatTensor([0])
    fenzi = torch.FloatTensor([0])
    for i_neighbor in neighbors:
        node = verts_packed[i_neighbor]

        alpha = torch.exp(-torch.norm(vertice-node))
        fenmu += alpha

        res = 1 - torch.abs(torch.sum((vertice-node) * verts_normals[i])) / torch.norm(vertice-node)
        fenzi += (alpha*(tex_gray[i] - tex_gray[i_neighbor])*res)

    delta_u[i] = fenzi/fenmu


#step 8 : calculate grad of curvature
t2 = time.time()

# pd1, pd2, pv1, pv2 = igl.principal_curvature(verts, tris)
# mean_curv = (pv1+pv2)/2
# mean_curv = np.clip(mean_curv, a_min=-0.05, a_max=0.05)

norm_w = 0.25 * inv_areas
mean_curv_norms = (L.mm(verts_packed) - verts_packed) * norm_w

print(time.time()-t2)

#step 9 : update vertice
eta = 0.2

#method1
# verts_refine = verts + ((eta*delta_u).view(-1).cpu().numpy()*0.15 - 0.01*(1-eta)*mean_curv)[:,np.newaxis]*(verts_normals.numpy())
# method 2
# verts_refine = verts + ((0.15*(eta*delta_u).view(-1).cpu().numpy())[:,np.newaxis]*verts_normals.numpy() - 0.002*mean_curv_norms.cpu().numpy())
#method3
mean_curv = mean_curv_norms.cpu().norm(1).numpy()
verts_refine = verts + ((1e-8*(eta*delta_u).view(-1).cpu().numpy() + 1e-8*(1-eta)*mean_curv)[:,np.newaxis])*verts_normals.numpy()

#更新之后保存看看：
with open('refined_final.obj', 'w') as f:
    # write vertices
    vertices = verts_refine.copy()
    for i in range(vertices.shape[0]):
        s = 'v {} {} {} \n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2])
        f.write(s)

    # write f: ver ind/ uv ind
    triangles_v = tris.copy()
    triangles_v += 1  # mesh lab start with 1
    triangles_t = triangles_v.copy()
    for i in range(triangles_v.shape[0]):
        s = 'f {}/{} {}/{} {}/{}\n'.format(triangles_v[i, 0], triangles_t[i, 0],
                                           triangles_v[i, 1], triangles_t[i, 1],
                                           triangles_v[i, 2], triangles_t[i, 2])
        f.write(s)

print(time.time()-start)
print('hhh')