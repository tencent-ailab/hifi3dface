# -*- coding: utf-8 -*-
"""
Our python implementation of a non-rigid variant of the iterative closest point algorithm. 
The original MATLAB implementation is from https://github.com/charlienash/nricp.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import spdiags
from scipy.sparse import kron
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve


class NRICPTool(object):
    def __init__(self):
        pass

    @staticmethod
    def nricp_shoulder(
        vertex_ref, vertex_me, tri, shoulder_idx, face_idx, idx_backhead, idx_contout
    ):
        tri = tri + 1  # python to  matlab

        Options_weight_up = 100000
        Options_weight_down = 100000
        Options_weight_mix = 1000
        Options_alpha = 10000
        Options_epsilon = 10e-4
        Options_maxIteration = 1

        nVertsSource = vertex_ref.shape[0]
        vertex_ref = vertex_ref.transpose()
        vertsTarget = vertex_me.transpose()
        nVertsSource = vertex_ref.shape[0]
        wVec_down = np.zeros((nVertsSource, 1))
        idx_backhead = [int(i) - 1 for i in idx_backhead]

        G = np.diag([1, 1, 1, 1])
        A = NRICPTool.triangulation2adjacency(tri.transpose())
        M = NRICPTool.adjacency2incidence(A).transpose()
        kron_M_G = kron(M, G)

        I = np.array([x for x in range(1, nVertsSource + 1)]).transpose()
        J = 4 * I
        t = np.hstack((I, I))
        t = np.hstack((t, I))
        t = np.hstack((t, I))

        t2 = np.hstack(([i - 3 for i in J], [i - 2 for i in J]))
        t2 = np.hstack((t2, [i - 1 for i in J]))
        t2 = np.hstack((t2, J))
        ver = np.reshape(vertex_ref, (-1, 1), order="F")
        data = np.vstack((ver, np.ones((nVertsSource, 1))))

        t = [i - 1 for i in t]
        t2 = [i - 1 for i in t2]
        data = np.squeeze(data)
        D = coo_matrix((data, (t, t2)), shape=(nVertsSource, 4 * nVertsSource))

        X = np.tile(np.vstack((np.eye(3), [0, 0, 0])), (nVertsSource, 1))

        wVec_shoulder = np.zeros((nVertsSource, 1))
        shoulder_idx = [int(i) - 1 for i in shoulder_idx]
        li = list(set(shoulder_idx).union(set(idx_backhead)))
        li = [int(i) for i in li]
        wVec_shoulder[li] = Options_weight_down
        W_shoulder = spdiags(wVec_shoulder.transpose(), 0, nVertsSource, nVertsSource)
        U_shoulder = vertex_ref

        wVec_face = np.zeros((nVertsSource, 1))
        face_idx = [int(i) - 1 for i in face_idx]
        wVec_face[face_idx, :] = Options_weight_up
        idx_contout = [int(i) - 1 for i in idx_contout]

        wVec_face[idx_contout, :] = Options_weight_mix
        W_face = spdiags(wVec_face.transpose(), 0, nVertsSource, nVertsSource)

        vertex_me = vertex_me.transpose()
        U_face = vertex_me
        nAlpha = 1
        print("* Performing non-rigid ICP...")

        for i in range(nAlpha):
            step = 0
            alpha = 10000
            oldX = 10 * X
            while np.linalg.norm(X - oldX) >= Options_epsilon:
                step = step + 1
                if step > Options_maxIteration:
                    break
                A0 = vstack((alpha * kron_M_G, coo_matrix(W_shoulder.dot(D))))

                A = vstack((A0, coo_matrix(W_face.dot(D))))

                s1 = M.shape[0]
                s2 = G.shape[0]
                zero = np.zeros((s1 * s2, 3))
                B0 = vstack((coo_matrix(zero), coo_matrix(W_shoulder.dot(U_shoulder))))
                B = vstack((B0, coo_matrix(W_face.dot(U_face))))

                oldX = X
                X = spsolve(A.transpose().dot(A), A.transpose().dot(B))

                one = D.dot(X)
                loss_down = np.abs(one.toarray() - vertex_ref)
                loss_down = np.sum(np.sum(loss_down[shoulder_idx, :]))
                loss_up = np.abs(one.toarray() - vertex_me)
                loss_up = np.sum(np.sum(loss_up[face_idx, :]))
                print(
                    "alpha: ", alpha, "  loss_up: ", loss_up, ",loss_down: ", loss_down
                )

        vertsTransformed = D.dot(X)
        vertsTransformed = vertsTransformed.transpose()
        return vertsTransformed

    @staticmethod
    def check_face_vertex(vertex, face):
        def check_size(a, vmin, vmax):
            a = np.array(a)
            if len(a) == 0:
                return -1
            if a.shape[0] > a.shape[1]:
                a = a.transpose()
            if a.shape[0] < 3 and a.shape[1] == 3:
                a = a.transpose()
            if a.shape[0] <= 3 and a.shape[1] >= 3 and sum(abs(a[:, 2])) == 0:
                a = a.transpose()
            if a.shape[0] < vmin or a.shape[0] > vmax:
                print("face or vertex is not of correct size")
            return a

        vertex = check_size(vertex, 2, 4)
        face = check_size(face, 3, 4)
        return vertex, face

    @staticmethod
    def triangulation2adjacency(face):
        tmp, face = NRICPTool.check_face_vertex([], face)
        f = face.transpose()
        t = np.hstack((f[:, 0], f[:, 0]))
        t = np.hstack((t, f[:, 1]))
        t = np.hstack((t, f[:, 1]))
        t = np.hstack((t, f[:, 2]))
        t = np.hstack((t, f[:, 2]))
        t = [int(i) - 1 for i in t]
        t2 = np.hstack((f[:, 1], f[:, 2]))
        t2 = np.hstack((t2, f[:, 0]))
        t2 = np.hstack((t2, f[:, 2]))
        t2 = np.hstack((t2, f[:, 0]))
        t2 = np.hstack((t2, f[:, 1]))
        t2 = [int(i) - 1 for i in t2]

        data = np.ones((len(t)))
        A = coo_matrix((data, (t2, t)), dtype=np.int)

        return A

    @staticmethod
    def adjacency2incidence(A):
        j = np.where(A.toarray())[0]
        i = np.where(A.toarray())[1]
        I = list(np.where(i <= j))
        i = i[I]
        j = j[I]

        n = len(i)
        nverts = A.shape[0]
        s = np.vstack((np.ones((n, 1)), -1 * np.ones((n, 1))))
        iS = np.hstack((range(n), range(n)))
        jS = np.hstack((i, j))
        data = np.hstack((np.ones((n)), -1 * np.ones((n))))

        Ic = coo_matrix((data, (iS, jS)), dtype=np.int)

        Ic = Ic.transpose()
        a = np.where(i == j)
        """
        if len(a) != 0:
            for t in a:
                Ic[i[t],t] = 1
        """
        return Ic
