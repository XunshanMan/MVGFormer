import numpy as np
from scipy.linalg import block_diag as block_diag

import torch

import time

class DictTree:
    """
    A efficient tree structure that discribes nodes in dictionaries. Use the 
    capital word NODE for nodes in this tree. All NODEs are dictionaries with
    three common keys - "name": name of the node; "index": the index of the node
    in node_list; "parent" the index of the node's parent.
    """
    def __init__(self, size, root):
        """
        size: <int> the number of nodes.
        root: <dict> the root NODE.
        """
        self.root = root
        self.size = size

        # the list of NODEs
        self.node_list = [{}]*size
        self.node_list[root["index"]] = root

        # the list of distal joint indices of bones.
        self.left_bones = []
        self.right_bones = []
        self.middle_bones = []

    def create_node(self, name, idx, parent):
        """
        Create a NODE and add it to the node_list.
        name, idx, parent respectively corespond to the "name", "index",
        "parent" keys.
        """
        node = {"name":name, "index":idx, "parent":parent}
        assert self.node_list[node["index"]] == {}, "Two nodes shares one index"
        self.node_list[node["index"]] = node

    def get_conv_mat(self):
        """
        Get the conversion matrix and its inverse.
        """
        conv_mat = np.zeros((self.size*3, self.size*3))
        for i in range(self.size):
            if i == self.root["index"]:
                conv_mat[0:3, 3*i:3*i+3] = np.eye(3)
            elif i < self.root["index"]:
                p = self.node_list[i]["parent"]
                conv_mat[3*i+3:3*i+6, 3*i:3*i+3] = np.eye(3)
                conv_mat[3*i+3:3*i+6, 3*p:3*p+3] = -np.eye(3)
            else:
                p = self.node_list[i]["parent"]
                conv_mat[3*i:3*i+3, 3*i:3*i+3] = np.eye(3)
                conv_mat[3*i:3*i+3, 3*p:3*p+3] = -np.eye(3)

        self.conv_J2B = conv_mat
        self.conv_B2J = np.linalg.inv(conv_mat)

    def draw_skeleton(self, ax, pts, joint_color="red", bone_color="auto"):
        """
        Draw human skeleton.
        :ax:          <matplotlib.axes> the axes to draw the skeleton
        :pts:         <numpy.ndarray> of n_joints x dims
        :joint_color: <string> the color to draw joints;
        :bone_color:  <string> the color to draw bones; "auto" means to use
        different colors to distinguish left, right and middle bones.
        :return:      <matplotlib.axes> the painted axes.
        """
        Nj = pts.shape[0]
        dim = pts.shape[1]
        bone_color_list = [bone_color] * (Nj-1)
        if bone_color == "auto":
            for i in self.left_bones:
                bone_color_list[i-1] = 'b'
            for i in self.right_bones:
                bone_color_list[i-1] = 'r'
            for i in self.middle_bones:
                bone_color_list[i-1] = 'gray'
        if dim == 2:
            for i in range(1, Nj):
                ax.plot(*[pt.reshape(2,) for pt in np.split(
                    pts[[i, self.node_list[i]["parent"]], :], 2, axis=1)],
                    color=bone_color_list[i-1])
            ax.scatter(*np.split(pts, 2, axis=1), color=joint_color)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        elif dim == 3:
            for i in range(1, Nj):
                ax.plot3D(*[pt.reshape(2,) for pt in np.split(
                    pts[[i, self.node_list[i]["parent"]], :], 3, axis=1)],
                    color=bone_color_list[i-1])
            # achieve equal visual lengths on three axes.
            ax.scatter3D(*np.split(pts, 3, axis=1), color=joint_color)
            extents = np.array(
                [getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:, 1] - extents[:, 0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            ax.auto_scale_xyz(*np.column_stack((centers - r, centers + r)))
            ax.view_init(elev=0, azim=0)
            tmp = [getattr(ax, 'set_{}ticks'.format(dim))([]) for dim in 'xyz']
        return ax

    def get_bl_mat(self, poses3D):
        """
        :pose3D: <numpy.ndarray> of n_frames x n_joints x 3, the 3D joint coordinates.
        :return: <numpy.ndarray> of n_frames x n_bones, the 3D bone length vector
        """
        n_frames = poses3D.shape[0]
        return np.linalg.norm((poses3D.reshape(n_frames, -1) @ self.conv_J2B.T)[:, 3:]\
                              .reshape(n_frames, -1, 3), axis=2).reshape(n_frames, -1)


def create_human_tree(data_type="human36m"):
    """
    create human tree structure according to data_type
    return a DictTree object.
    """
    if data_type == "human36m":
        human_tree = DictTree(17, {"name":"Hip", "index":0})
        human_tree.create_node("RHip", 2, parent=0)
        human_tree.create_node("RKnee", 1, parent=2)
        human_tree.create_node("RFoot", 6, parent=1)
        human_tree.create_node("LHip", 3, parent=0)
        human_tree.create_node("LKnee", 4, parent=3)
        human_tree.create_node("LFoot", 5, parent=4)
        human_tree.create_node("Spine", 7, parent=0)
        human_tree.create_node("Thorax", 8, parent=7)
        human_tree.create_node("Neck", 16, parent=8)
        human_tree.create_node("Head", 9, parent=16)
        human_tree.create_node("LShoulder", 13, parent=8)
        human_tree.create_node("LElbow", 14, parent=13)
        human_tree.create_node("LWrist", 15, parent=14)
        human_tree.create_node("RShoulder", 12, parent=8)
        human_tree.create_node("RElbow", 11, parent=12)
        human_tree.create_node("RWrist", 10, parent=11)
        human_tree.left_bones = [3, 4, 5, 13, 14, 15]
        human_tree.right_bones = [2, 1, 6, 12, 11, 10]
        human_tree.middle_bones = [7, 8, 16, 9]
    elif data_type == "totalcapture":
        human_tree = DictTree(16, {"name":"Hip", "index":0})
        human_tree.create_node("RHip", 2, parent=0)
        human_tree.create_node("RKnee", 1, parent=2)
        human_tree.create_node("RFoot", 6, parent=1)
        human_tree.create_node("LHip", 3, parent=0)
        human_tree.create_node("LKnee", 4, parent=3)
        human_tree.create_node("LFoot", 5, parent=4)
        human_tree.create_node("Spine", 7, parent=0)
        human_tree.create_node("Thorax", 8, parent=7)
        human_tree.create_node("Head", 9, parent=8)
        human_tree.create_node("LShoulder", 13, parent=8)
        human_tree.create_node("LElbow", 14, parent=13)
        human_tree.create_node("LWrist", 15, parent=14)
        human_tree.create_node("RShoulder", 12, parent=8)
        human_tree.create_node("RElbow", 11, parent=12)
        human_tree.create_node("RWrist", 10, parent=11)
        human_tree.left_bones = [3, 4, 5, 13, 14, 15]
        human_tree.right_bones = [2, 1, 6, 12, 11, 10]
        human_tree.middle_bones = [7, 8, 9]
    elif data_type == "cmupanoptic":
        human_tree = DictTree(15, {"name":"Hip", "index":0})
        LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
           [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]
        
        for j,[a,b] in enumerate(LIMBS15):
            human_tree.create_node(f"Bone-{j}", b, parent=a)
        human_tree.left_bones = [9,10,11,12,13,14]
        human_tree.right_bones = [3,4,5,6,7,8]
        human_tree.middle_bones = [1,0,2]        
        
    human_tree.get_conv_mat()

    return human_tree


def get_inner_mat(u, v):
    return np.array([[1, 0, -u], [0, 1, -v], [-u, -v, u**2+v**2]])

def get_inner_mat_torch(u, v, device):
    method = 'gradients'
    
    if method == 'origin':
        return torch.Tensor([[1, 0, -u], [0, 1, -v], [-u, -v, u**2+v**2]]).to(device)
    elif method == 'gradients':
        '''
        with gradients
        '''
        mat = torch.zeros(3,3).float().to(device)
        mat[0,0] = 1
        mat[1,1] = 1
        mat[0,2] = -u
        mat[1,2] = -v
        mat[2,0] = -u
        mat[2,1] = -v
        mat[2,2] = u**2 + v**2
        
        # check grad
        # print(mat)
        return mat

'''
A batch version to speed up with GPU acceleration
'''
def Pose3D_inference_torch_batch(n_cams, human_tree, poses_2d, confidences, lengths, Projections,
        method, n_step):
    """
    Batch version. Faster.
    
    [Torch Version with Auto Differentiate]
    The main procedure of structural triangulation & step contraint algorithm
    input params:
    : n_cams :      <int> the number of cameras
    : human_tree :  <DictTree> the tree of human structure.
    : poses_2d :    <numpy.ndarray> Batch x n_cams x n_joints x 2. The 2D joint estimates.
    : confidences : <numpy.ndarray> Batch x n_cams x n_joints. The confidences between cameras.
    : lengths :     <numpy.ndarray> Batch x (n_joints - 1) x 1. The ground truth lengths.
    : Projections : <numpy.ndarray> of Batch x n_cams x 3 x 4. The concatenation of
    projection matrices.
    """
    t1 = time.time()
    device = poses_2d.device
    # Number of joints 
    Nj = human_tree.size
    batch_size = poses_2d.shape[0]

    # Transformation matrix from bone to joint
    G = human_tree.conv_B2J
    G = torch.from_numpy(G).float().to(device)

    # TODO: Begin here, Batchilize the following matrices calculation
    tmp = []
    temp_batch = 0
    for j in range(n_cams):
        tmp.append(Projections[temp_batch, j, :, 0:3])
    # KR_diag = [np.concatenate(tmp, axis=0)]*Nj
    KR_diag = [torch.cat(tmp,axis=0)]*Nj
    KR = torch.block_diag(*KR_diag)   
    # Copy BATCH Size Time for KR Matrices
    KR_batch = KR.expand(batch_size,-1,-1)
    
    P = torch.zeros((Nj * n_cams * 3, Nj * n_cams * 3)).to(device)
    t2 = time.time()

    if confidences is None:
        confidences = torch.ones((n_cams, Nj)).to(device) / n_cams
    for i in range(Nj):
        for j in range(n_cams):
            conf = confidences[j, i]
            P[3*(i*n_cams + j):3*(i*n_cams + j)+3,
              3*(i*n_cams + j):3*(i*n_cams + j)+3] = \
              conf * get_inner_mat_torch(poses_2d[j, i, 0], poses_2d[j, i, 1], device)
    t3 = time.time()

    # Temp test cpu loading
    # No obvious diff is seen
    # t3_start = time.time()
    # P = torch.zeros((Nj * n_cams * 3, Nj * n_cams * 3))
    # if confidences is None:
    #     confidences = torch.ones((n_cams, Nj)) / n_cams
    # for i in range(Nj):
    #     for j in range(n_cams):
    #         conf = confidences[j, i]
    #         P[3*(i*n_cams + j):3*(i*n_cams + j)+3,
    #           3*(i*n_cams + j):3*(i*n_cams + j)+3] = \
    #           conf * get_inner_mat_torch(poses_2d[j, i, 0], poses_2d[j, i, 1], device='cpu')
    # t3_end = time.time()
    # print('cpu_t3:', t3_end-t3_start)
    ###########################



    D = 2 * KR.T @ P @ KR

    Irow = torch.cat((torch.eye(3),)*Nj, axis=1).to(device)
    Mrow = Irow @ D
    TrLam = Mrow @ Irow.T
    Mrow = Mrow[:, 3:]
    TrM_inv = torch.linalg.inv(TrLam)

    tmp = []
    for j in range(n_cams):
        tmp.append(Projections[j, :, 3:4])
    KRT = -torch.cat(tuple(tmp)*Nj, axis=0)
    m = 2 * (KRT.T @ P @ KR).T
    t4 = time.time()

    Q = torch.cat((-TrM_inv @ Mrow @ G[3:, 3:], torch.eye(Nj*3-3).to(device)), axis=0)
    p = torch.cat((-TrM_inv @ Irow @ m, torch.zeros((Nj*3-3, 1)).to(device)), axis=0)

    A = Q.T @ G.T @ D @ G @ Q
    beta = (p.T @ G.T @ D @ G @ Q + m.T @ G @ Q).T

    A_inv = torch.linalg.inv(A)
    b0 = A_inv @ beta
    D31 = torch.zeros((3*Nj-3, Nj-1)).to(device)
    for i in range(Nj-1):
        D31[3*i:3*i+3, i:i+1] = torch.ones((3, 1)).to(device)
    t5 = time.time()

    if method == "Lagrangian":
        b = Lagrangian_method(A, beta, b0, n_step, Nj, lengths, D31)
    elif method == "ST":
        b = ST_SCA_torch(A_inv, beta, Nj, b0, lengths, D31, n_step, device)
    elif method == "LS":
        b = b0
    else:
        print(f"Method {method} not completed yet.")
        exit(-1)
    t6 = time.time()

    x0 = -TrM_inv @ (Mrow @ G[3:, 3:] @ b - Irow @ m)
    X = G @ torch.cat((x0, b), axis=0)
    t7 = time.time()

    print('t2:', t2-t1)
    print('t3:', t3-t2)
    print('t4:', t4-t3)
    print('t5:', t5-t4)
    print('t6:', t6-t5)
    print('t7:', t7-t6)

    return X.reshape(Nj, 3)


def Pose3D_inference_torch(n_cams, human_tree, poses_2d, confidences, lengths, Projections,
        method, n_step):
    """
    ONE PERSON EACH TIME
    
    [Torch Version with Auto Differentiate]
    The main procedure of structural triangulation & step contraint algorithm
    input params:
    : n_cams :      <int> the number of cameras
    : human_tree :  <DictTree> the tree of human structure.
    : poses_2d :    <numpy.ndarray> n_cams x n_joints x 2. The 2D joint estimates.
    : confidences : <numpy.ndarray> n_cams x n_joints. The confidences between cameras.
    : lengths :     <numpy.ndarray> (n_joints - 1) x 1. The ground truth lengths.
    : Projections : <numpy.ndarray> of n_cams x 3 x 4. The concatenation of
    projection matrices.
    """
    device = poses_2d.device
    # Number of joints 
    Nj = human_tree.size

    # Transformation matrix from bone to joint
    G = human_tree.conv_B2J
    G = torch.from_numpy(G).float().to(device)

    tmp = []
    for j in range(n_cams):
        tmp.append(Projections[j, :, 0:3])
    # KR_diag = [np.concatenate(tmp, axis=0)]*Nj
    KR_diag = [torch.cat(tmp,axis=0)]*Nj
    
    # KR = block_diag(*KR_diag)
    KR = torch.block_diag(*KR_diag)
    P = torch.zeros((Nj * n_cams * 3, Nj * n_cams * 3)).to(device)

    if confidences is None:
        confidences = torch.ones((n_cams, Nj)).to(device) / n_cams
    for i in range(Nj):
        for j in range(n_cams):
            conf = confidences[j, i]
            P[3*(i*n_cams + j):3*(i*n_cams + j)+3,
              3*(i*n_cams + j):3*(i*n_cams + j)+3] = \
              conf * get_inner_mat_torch(poses_2d[j, i, 0], poses_2d[j, i, 1], device)
    D = 2 * KR.T @ P @ KR

    Irow = torch.cat((torch.eye(3),)*Nj, axis=1).to(device)
    Mrow = Irow @ D
    TrLam = Mrow @ Irow.T
    Mrow = Mrow[:, 3:]
    TrM_inv = torch.linalg.inv(TrLam)

    tmp = []
    for j in range(n_cams):
        tmp.append(Projections[j, :, 3:4])
    KRT = -torch.cat(tuple(tmp)*Nj, axis=0)
    m = 2 * (KRT.T @ P @ KR).T

    Q = torch.cat((-TrM_inv @ Mrow @ G[3:, 3:], torch.eye(Nj*3-3).to(device)), axis=0)
    p = torch.cat((-TrM_inv @ Irow @ m, torch.zeros((Nj*3-3, 1)).to(device)), axis=0)

    A = Q.T @ G.T @ D @ G @ Q
    beta = (p.T @ G.T @ D @ G @ Q + m.T @ G @ Q).T

    A_inv = torch.linalg.inv(A)
    b0 = A_inv @ beta
    D31 = torch.zeros((3*Nj-3, Nj-1)).to(device)
    for i in range(Nj-1):
        D31[3*i:3*i+3, i:i+1] = torch.ones((3, 1)).to(device)

    if method == "Lagrangian":
        b = Lagrangian_method(A, beta, b0, n_step, Nj, lengths, D31)
    elif method == "ST":
        b = ST_SCA_torch(A_inv, beta, Nj, b0, lengths, D31, n_step, device)
    elif method == "LS":
        b = b0
    else:
        print(f"Method {method} not completed yet.")
        exit(-1)

    x0 = -TrM_inv @ (Mrow @ G[3:, 3:] @ b - Irow @ m)
    X = G @ torch.cat((x0, b), axis=0)
    return X.reshape(Nj, 3)

def Pose3D_inference(n_cams, human_tree, poses_2d, confidences, lengths, Projections,
        method, n_step):
    """
    The main procedure of structural triangulation & step contraint algorithm
    input params:
    : n_cams :      <int> the number of cameras
    : human_tree :  <DictTree> the tree of human structure.
    : poses_2d :    <numpy.ndarray> n_cams x n_joints x 2. The 2D joint estimates.
    : confidences : <numpy.ndarray> n_cams x n_joints. The confidences between cameras.
    : lengths :     <numpy.ndarray> (n_joints - 1) x 1. The ground truth lengths.
    : Projections : <numpy.ndarray> of n_cams x 3 x 4. The concatenation of
    projection matrices.
    """
    # Number of joints 
    Nj = human_tree.size

    # Transformation matrix from bone to joint
    G = human_tree.conv_B2J

    tmp = []
    for j in range(n_cams):
        tmp.append(Projections[j, :, 0:3])
    KR_diag = [np.concatenate(tmp, axis=0)]*Nj
    KR = block_diag(*KR_diag)
    P = np.zeros((Nj * n_cams * 3, Nj * n_cams * 3))

    if confidences is None:
        confidences = np.ones((n_cams, Nj)) / n_cams
    for i in range(Nj):
        for j in range(n_cams):
            conf = confidences[j, i]
            P[3*(i*n_cams + j):3*(i*n_cams + j)+3,
              3*(i*n_cams + j):3*(i*n_cams + j)+3] = \
              conf * get_inner_mat(poses_2d[j, i, 0], poses_2d[j, i, 1])
    D = 2 * KR.T @ P @ KR

    Irow = np.concatenate((np.eye(3),)*Nj, axis=1)
    Mrow = Irow @ D
    TrLam = Mrow @ Irow.T
    Mrow = Mrow[:, 3:]
    TrM_inv = np.linalg.inv(TrLam)

    tmp = []
    for j in range(n_cams):
        tmp.append(Projections[j, :, 3:4])
    KRT = -np.concatenate(tuple(tmp)*Nj, axis=0)
    m = 2 * (KRT.T @ P @ KR).T

    Q = np.concatenate((-TrM_inv @ Mrow @ G[3:, 3:], np.eye(Nj*3-3)), axis=0)
    p = np.concatenate((-TrM_inv @ Irow @ m, np.zeros((Nj*3-3, 1))), axis=0)

    A = Q.T @ G.T @ D @ G @ Q
    beta = (p.T @ G.T @ D @ G @ Q + m.T @ G @ Q).T

    A_inv = np.linalg.inv(A)
    b0 = A_inv @ beta
    D31 = np.zeros((3*Nj-3, Nj-1))
    for i in range(Nj-1):
        D31[3*i:3*i+3, i:i+1] = np.ones((3, 1))

    if method == "Lagrangian":
        b = Lagrangian_method(A, beta, b0, n_step, Nj, lengths, D31)
    elif method == "ST":
        b = ST_SCA(A_inv, beta, Nj, b0, lengths, D31, n_step)
    elif method == "LS":
        b = b0
    else:
        print(f"Method {method} not completed yet.")
        exit(-1)

    x0 = -TrM_inv @ (Mrow @ G[3:, 3:] @ b - Irow @ m)
    X = G @ np.concatenate((x0, b), axis=0)
    return X.reshape(17, 3)


def Lagrangian_method(A, e, b0, n_iter, Nj, lengths, D31):
    """
    Implementation of Lagrangian Algorithm for constrained HPE problem.
    """
    b = b0
    lam = np.zeros((Nj-1, 1))
    alpha = 0.000000002
    beta = 0.5
    for k in range(n_iter):
        Dh = D31.T @ np.diag(b.reshape(-1,))
        bn = b - alpha * (A @ b - e + 2 * Dh.T @ lam)
        hk = np.square(np.linalg.norm(b.reshape(-1, 3), axis=1).reshape(-1, 1))\
            - np.square(lengths)
        lamn = lam + beta * hk
        # if np.linalg.norm(b - bn) < 1:
        #     break
        b = bn
        lam = lamn

    return b


def ST_SCA(A_inv, beta, Nj, b0, lengths, D31, n_step):
    """
    Structural Triangulation with step contrain algorithm. When n_step == 1,
    this is pure ST without SCA.
    """
    b = b0
    Inv = A_inv
    for i in range(n_step):
        start_len = np.linalg.norm(b.reshape(-1, 3), axis=1).reshape(-1, 1)
        target_len = (start_len * (n_step - i - 1) + lengths) / (n_step - i)
        Db = np.diag(b.reshape(-1,))
        tmp_inv = np.linalg.inv(D31.T @ Db @ Inv @ Db @ D31 + 0 * np.eye(Nj-1))
        lam = tmp_inv @ (np.square(start_len) - np.square(target_len)) / 4
        D_lambda = np.diag(np.concatenate((2 * lam, )*3, axis=1).reshape(-1,))

        Inv = (np.eye(3*Nj-3) - Inv @ D_lambda) @ Inv
        b = Inv @ beta
    return b

def ST_SCA_torch(A_inv, beta, Nj, b0, lengths, D31, n_step, device):
    """
    Structural Triangulation with step contrain algorithm. When n_step == 1,
    this is pure ST without SCA.
    """
    b = b0
    Inv = A_inv
    for i in range(n_step):
        start_len = torch.linalg.norm(b.reshape(-1, 3), axis=1).reshape(-1, 1)
        target_len = (start_len * (n_step - i - 1) + lengths) / (n_step - i)
        Db = torch.diag(b.reshape(-1,))
        tmp_inv = torch.linalg.inv(D31.T @ Db @ Inv @ Db @ D31 + 0 * torch.eye(Nj-1).to(device))
        lam = tmp_inv @ (torch.square(start_len) - torch.square(target_len)) / 4
        D_lambda = torch.diag(torch.cat((2 * lam, )*3, axis=1).reshape(-1,))   # 42 x 42

        Inv = (torch.eye(3*Nj-3).to(device) - Inv @ D_lambda) @ Inv
        b = Inv @ beta
    return b

if __name__ == '__main__':
    # a small optimization to see if we can differentiate the small function
    pass