import tkinter as tk
from collections import deque
import time
from tkinter.filedialog import askopenfilename, asksaveasfilename
from queue import Queue
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d
from scipy.spatial import ConvexHull

from stl import mesh
from rtree import index
import threading
import numpy as np
import scipy.optimize

def turnIt():
    root = tk.Tk()

    # critical angle; facets more overhung than this angle will require support;
    # an angle of 0 means _all_ faces with an outward normal that points at all downward
    # would require support:
    CRITICAL_ANGLE_DEG = 35.0
    DEFAULT_CRITICAL_ANGLE_DEG = 40.0
    CRITICAL_ANGLE_RAD = CRITICAL_ANGLE_DEG * np.pi / 180.0

    OPTIMIZE_STATE_STOPPED = 0
    OPTIMIZE_STATE_RUNNING = 1
    OPTIMIZE_STATE_STOPPING = 2

    MIN_VARIANCE = 0.005

    WRITE_TEXT_CMD = 'WRITE'
    NEW_BEST_ORIENTATION_CMD = 'NEWBEST'
    FINISHED_CMD = 'FINISHED'

    MIN_REDRAW_TIME = 1.0
    N_RAND_VECS = 35

    class ThreadInt:
        def __init__(self, i):
            self.lock = threading.RLock()
            self.set(i)

        def set(self, i):
            with self.lock:
                self.i = i

        def get(self):        
            with self.lock:
                reti = self.i

            return reti

    def get_facet_XY_bounding_box(obj, i_facet):
        return (obj.vectors[i_facet, :, 0].min(), obj.vectors[i_facet, :, 1].min(),
                obj.vectors[i_facet, :, 0].max(), obj.vectors[i_facet, :, 1].max())

    def get_proj_area(obj, i_facet):
        facet = obj.vectors[i_facet]
        v0 = facet[2] - facet[0]
        v1 = facet[1] - facet[0]
        return np.abs(v0[1] * v1[0] - v0[0] * v1[1])

    def is_XY_pt_in_XY_projection(P, obj, i_facet, **kw):
        facet = obj.vectors[i_facet]
        A = facet[0]
        B = facet[1]
        C = facet[2]
        v0 = (C - A)[0:2]
        v1 = (B - A)[0:2]
        v2 = (P - A)[0:2]

        dot00 = kw['dot00'][i_facet]
        dot11 = kw['dot11'][i_facet]
        dot01 = kw['dot01'][i_facet]
        dot02 = v0.dot(v2)
        dot12 = v1.dot(v2)
        invDenom = kw['invDenom'][i_facet]

        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        if u > -1.0E-6 and v > -1.0E6 and u + v <= 1.000001:
            height = A[2] + u * (C[2] - A[2]) + v * (B[2] - A[2])

            return True, height
        else:
            return False, 0.0

    def get_support_height(**kw):

        support_bottom = kw['minZ']
        r_tree = kw['rtree']
        pt = kw['point']

        #print('top: {:.2f}'.format(pt[2]))
        for possible_support_facet_i in r_tree.intersection(tuple(pt[0:2])):
            is_in, height = is_XY_pt_in_XY_projection(pt, kw['obj'], possible_support_facet_i, dot00=kw['dot00'],
                                                      dot11=kw['dot11'], dot01=kw['dot01'], invDenom=kw['invDenom'])
            if is_in:
                pass
                #print('possible support facet {} has a support height {:.8f}'.format(possible_support_facet_i, height))
            if is_in and height > support_bottom and height < pt[2] - 1.0E-4:
                support_bottom = height
                #print('adjusting support bottom to {:.8f}'.format(support_bottom))

        return pt[2] - support_bottom    


    # determine the volume of support material necessary for a given object:
    def get_amount_support(obj):
        # create an R-tree of the XY-plane projections of the upward-facing facets;
        # these facets represent potential support facets:
        minZ = obj.vectors[:, :, 2].min()
        N_facets = obj.normals.shape[0]

        As = obj.vectors[:, 0, 0:2]
        Bs = obj.vectors[:, 1, 0:2]
        Cs = obj.vectors[:, 2, 0:2]
        v0s = Cs - As
        v1s = Bs - As
        dot00s = (v0s * v0s).sum(axis=1)
        dot11s = (v1s * v1s).sum(axis=1)
        dot01s = (v0s * v1s).sum(axis=1)
        denoms = dot00s * dot11s - dot01s * dot01s

        a = obj.vectors[:, 0]
        b = obj.vectors[:, 1]
        c = obj.vectors[:, 2]
        normals = np.cross(b - a, c - a)
        normals = normals / (np.sqrt((obj.normals**2).sum(axis=1)) + 1.0E-9)[:, np.newaxis]

        up_facets = np.argwhere(np.logical_and((normals[:, 2] > 0.0), (np.abs(denoms) > 1.0E-8)))
        denoms = np.where(np.abs(denoms) > 1.0E-8, denoms, 1.0)
        invDenoms = 1.0 / denoms

        up_index = index.Index()
        for i_facet in up_facets:
            bbox = get_facet_XY_bounding_box(obj, i_facet)
            up_index.insert(i_facet, bbox)

        crit_z = -np.sin(CRITICAL_ANGLE_RAD)
        need_support_facets = np.argwhere(normals[:, 2] < crit_z).flatten()
        print('facets needing support:', need_support_facets.shape[0])
        #for i_facet in need_support_facets:
        #    print('facet {}: {}, normal: {}'.format(i_facet, obj.vectors[i_facet], normals[i_facet]))

        # for each facet needing support, determine the distance down to support:
        support_vol = 0.0

        supports = []
        for i_facet in need_support_facets:
            pts = obj.vectors[i_facet]
            # get the support height at each corner of the facet:
            support_heights = []
            for ipt, pt in enumerate(pts):            
                #print('getting support height for facet {}, pt {} = {}'.format(i_facet, ipt, pt))
                height = get_support_height(obj=obj, point=pt, rtree=up_index, minZ=minZ, dot00=dot00s,
                                            dot11=dot11s, dot01=dot01s, invDenom=invDenoms)
                support_heights.append(height)

            if sum(support_heights) > 1.0E-4:
                supports.append([i_facet, support_heights])
            addl_support_vol = get_proj_area(obj, i_facet) * sum(support_heights) / 3.0
            #print('facet:', pts, 'support vol:', addl_support_vol)
            # the volume of support needed to suspend this facet is the XY-plane-projected area
            # of the facet times the average needed support height:
            support_vol += addl_support_vol

        return support_vol, supports    

    def get_biggest_hull_normals(obj, cutoff):
        vecs = obj.vectors[:, :, :]
        vecs = vecs.reshape((obj.vectors.shape[0] * obj.vectors.shape[1], 3))
        print('hey:', vecs.shape)
        hull = ConvexHull(vecs)
        def tri_area(tri):
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            c = np.cross(v1, v2)
            return 0.5 * np.sqrt(c.dot(c))
        areas = np.array([tri_area(vecs[simplex]) for simplex in hull.simplices])
        areas /= areas.sum()
        def norm_vec(tri):
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            c = np.cross(v1, v2)
            c /= np.sqrt(c.dot(c))
            return c
        normals = []
        for i, area in enumerate(areas):
            if area >= cutoff:
                normals.append(norm_vec(vecs[hull.simplices[i]]))

        return normals

    def fibonacci_sphere(samples=1,randomize=True):
        rnd = 1.
        if randomize:
            rnd = np.random.rand() * samples

        points = []
        offset = 2./samples
        increment = np.pi * (3. - np.sqrt(5.));

        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2);
            r = np.sqrt(1 - pow(y,2))

            phi = ((i + rnd) % samples) * increment

            x = np.cos(phi) * r
            z = np.sin(phi) * r

            points.append(np.array([x,y,z]))

        return points

    class OptException(Exception): 
        pass

    class Optimizer:
        def __init__(self, start_obj, msg_queue, opt_state, critical_ang):
            self.start_obj = start_obj
            self.msg_queue = msg_queue
            self.opt_state = opt_state
            self.critical_ang = critical_ang
            self.basisV1 = self.basisV2 = self.basisV3 = None
            self.cost_vals = deque(np.arange(10) + 1, 8)

        # return a rotated version of the object, rotated
        # such that the vector v points along +z after the rotation;
        # this leaves a final rotation around z ambiguous, but for our purposes
        # it is unimportant:
        def get_rotated_obj(self, obj, v):
            v_norm = v / np.sqrt(v.dot(v))

            # get the cross-product of the z-axis and v:
            z = np.array([0, 0, 1.0])
            c = np.cross(z, v_norm)
            c_norm = c / np.sqrt(c.dot(c))
            rot_ang = np.arccos(v_norm.dot(z))
            new_mesh = mesh.Mesh(obj.data.copy())
            new_mesh.rotate(c_norm, rot_ang)

            return new_mesh        

        def msg(self, m):
            self.msg_queue.put([WRITE_TEXT_CMD, m + '\n'])

        def new_best_orientation(self, cost, vec, newobj, supports):
            self.min_cost = cost
            self.best_vec = vec        
            self.best_obj = newobj
            self.supports = supports
            self.msg('New minimum support amount: {:.0f} mm^3'.format(cost))        
            self.msg_queue.put([NEW_BEST_ORIENTATION_CMD, newobj, supports])

        def finished(self):
            self.msg('Optimization finished.')
            self.opt_state.set(OPTIMIZE_STATE_STOPPED)
            self.msg_queue.put([FINISHED_CMD, None])

        def get_basis_vecs(self, v):
            self.basisV1 = v[:]
            self.basisV1 /= np.sqrt(self.basisV1.dot(self.basisV1))
            if np.abs(self.basisV1[2]) > 0.5:
                w = np.array([1.0, 0.0, 0.0])
            else:
                w = np.array([0.0, 0.0, 1.0])
            self.basisV2 = np.cross(self.basisV1, w)
            self.basisV2 /= np.sqrt(self.basisV2.dot(self.basisV2))
            self.basisV3 = np.cross(self.basisV1, self.basisV2)        

        def get_vec(self, v):
            vec = self.basisV1 + v[0] * self.basisV2 + v[1] * self.basisV3
            vec /= np.sqrt(vec.dot(vec))
            return vec

        def optimize(self):
            N_TEST_VECS = 40
            DEV = 0.2
            test_vecs = [np.array([0, 0, 1.0])]
            test_vecs += fibonacci_sphere(N_RAND_VECS)
            big_hull_normals = get_biggest_hull_normals(self.start_obj, 0.05)
            test_vecs += big_hull_normals
            self.msg('Number of big faces on convex hull: {}'.format(len(big_hull_normals)))        
            self.msg('Number of test vectors: {}'.format(len(test_vecs)))

            self.min_cost = None
            costs = []
            for ivec, vec in enumerate(test_vecs):
                self.get_basis_vecs(vec)
                self.msg('Trying test vector #{}/{}: ({:.3f}, {:.3f}, {:.3f})'.format(ivec + 1, len(test_vecs), vec[0], vec[1], vec[2]))
                try:
                    v, cost, rotobj, supports = self.cost_func([0, 0.])
                except OptException as e:
                    self.finished()
                    return
                costs.append(cost)
                self.msg('support volume: {:.0f} mm^3'.format(cost))
                if self.min_cost is None or cost < self.min_cost:
                    self.new_best_orientation(cost, vec, rotobj, supports)
                if cost < 1.0E-2:
                    self.msg('Found orientation with near-zero support.')
                    self.finished()
                    return

            costs = np.array(costs)
            min_vec = test_vecs[np.argmin(costs)]
            self.get_basis_vecs(min_vec)
            initial_simplex = DEV * (np.random.rand(3, 2) - 0.5)
            try:
                result = scipy.optimize.minimize(self.min_func, np.array([0.0, 0.0]),
                                                 method='Nelder-Mead', 
                                                 options={'initial_simplex': initial_simplex,
                                                          'fatol': 10.0})
                x = result.x
            except OptException as e:
                self.finished()

        def cost_func(self, x):
            vec = self.get_vec(x)    
            rot_obj = self.get_rotated_obj(self.start_obj, vec)
            amount_support, supports = get_amount_support(rot_obj)
            self.cost_vals.append(amount_support)
            if self.opt_state.get() != OPTIMIZE_STATE_RUNNING:
                self.msg('Optimization halted.')
                raise OptException()
            return vec, amount_support, rot_obj, supports

        def min_func(self, x):
            vec, amount_support, rot_obj, supports = self.cost_func(x)
            self.msg('Support vol: {:.0f} mm^3'.format(amount_support))
            if self.min_cost is None or amount_support < self.min_cost:
                self.new_best_orientation(amount_support, vec, rot_obj, supports)

            if amount_support < 1.0E-2:
                self.msg('Found orientation with near-zero support.')            
                raise OptException()

            fvals = np.array(self.cost_vals)
            sdev = fvals.std()
            mean = fvals.mean()
            self.msg('Variance of last 5 costs: {:.1f}%'.format(sdev / mean * 100.0))
            if sdev/mean < MIN_VARIANCE:
                self.msg('Variance fell below minimum.')
                raise OptException()

            return amount_support

    class App(tk.Frame):
        def __init__(self):
            super().__init__()
            self.initUI()
            self.obj3d = None
            self.obj_waiting_to_plot = self.supports_waiting_to_plot = None
            self.optimize_state = ThreadInt(OPTIMIZE_STATE_STOPPED)
            self.last_draw_finished = 0.0
            self.best_obj = None

        def optimize(self):
            if self.optimize_state.get() != OPTIMIZE_STATE_STOPPED:
                self.text_output.insert(tk.END, 'Optimization already running.\n')
            else:
                self.text_output.insert(tk.END, 'Starting optimization.\n')            
                self.start_optimization()

        def stop(self):
            if self.optimize_state.get() == OPTIMIZE_STATE_RUNNING:
                self.text_output.insert(tk.END, 'Stopping optimization.\n')            
                self.optimize_state.set(OPTIMIZE_STATE_STOPPING)

        def start_optimization(self):
            self.msg_queue = Queue()
            self.read_crit_ang()
            self.optimize_state.set(OPTIMIZE_STATE_RUNNING)
            self.opt = Optimizer(self.obj3d, self.msg_queue, self.optimize_state, self.critical_ang)
            thread = threading.Thread(target=self.opt.optimize)
            thread.start()
            self.after(100, self.listen_for_messages)

        def listen_for_messages(self):
            if self.msg_queue.empty():
                self.after(100, self.listen_for_messages)
            else:
                while not self.msg_queue.empty():
                    print('queue not empty')
                    obj = self.msg_queue.get()
                    cmd = obj[0]
                    data = obj[1]
                    if cmd == WRITE_TEXT_CMD:
                        self.text_output.insert(tk.END, data)
                        self.text_output.see('end')
                    elif cmd == NEW_BEST_ORIENTATION_CMD:
                        self.best_obj = obj[1]
                        self.obj_waiting_to_plot = obj[1]
                        self.supports_waiting_to_plot = obj[2]
                        t = time.time()
                        if t - self.last_draw_finished > MIN_REDRAW_TIME:
                            self.plot_obj_support()
                    elif cmd == FINISHED_CMD:
                        self.optimize_state.set(OPTIMIZE_STATE_STOPPED)
                        if self.obj_waiting_to_plot is not None:
                            self.plot_obj_support()

                        if self.best_obj is not None and self.out_STL_file.get().strip() != '':
                            self.text_output.insert(tk.END, 'Saving best orientation to {}\n'.format(self.out_STL_file.get()))
                            self.best_obj.save(self.out_STL_file.get())
                        return
                self.after(100, self.listen_for_messages)            

        def load_STL(self):
            self.text_output.delete(1.0,tk.END)
            self.obj3d = mesh.Mesh.from_file(self.in_STL_file.get())
            self.text_output.insert(tk.END, 'Loaded {}\n'.format(self.in_STL_file.get()))                

        def plot_obj_support(self):
            # Create a new plot
            obj = self.obj_waiting_to_plot
            supports = self.supports_waiting_to_plot
            self.fig.clear()
            axes = mplot3d.Axes3D(self.fig)

            def get_poly_shades(polys, inv):
                v = inv / np.sqrt(inv.dot(inv))
                a = polys[:,0]
                b = polys[:,1]
                c = polys[:,2]
                v1 = b-a
                v2 = c-a
                crosses = np.cross(v1,v2)
                crosses /= np.sqrt((crosses*crosses).sum(axis=1))[:, np.newaxis]
                dots = np.sum(crosses * v[np.newaxis, :], axis=1)        
                shade = 0.5 * dots + 0.5

                return shade

            print(obj.vectors.shape)

            cols = np.zeros((obj.vectors.shape[0], 3))
            shades = get_poly_shades(obj.vectors, np.array([1.0, 1.0, 1.0]))
            cols[:, 0] = 0.7 * shades + 0.3
            cols[:, 1] = 0.7 * shades + 0.3
            cols[:, 2] = 0
            obj_cols = cols        

            supp_polys = np.zeros((len(supports), 3, 4, 3))

            if len(supports) != 0:
                support_facets = np.array([support[0] for support in supports])
                support_heights = np.array([support[1] for support in supports])
                support_vecs = np.zeros((len(supports), 3, 3))
                support_vecs[:, :, 2] = -support_heights
                supp_polys[:, 0, 0] = obj.vectors[support_facets][:, 0]
                supp_polys[:, 0, 1] = obj.vectors[support_facets][:, 1]    
                supp_polys[:, 0, 2] = obj.vectors[support_facets][:, 1] + support_vecs[:, 1]       
                supp_polys[:, 0, 3] = obj.vectors[support_facets][:, 0] + support_vecs[:, 0]        
                supp_polys[:, 1, 0] = obj.vectors[support_facets][:, 1]
                supp_polys[:, 1, 1] = obj.vectors[support_facets][:, 2]    
                supp_polys[:, 1, 2] = obj.vectors[support_facets][:, 2] + support_vecs[:, 2]               
                supp_polys[:, 1, 3] = obj.vectors[support_facets][:, 1] + support_vecs[:, 1]       
                supp_polys[:, 2, 0] = obj.vectors[support_facets][:, 2]
                supp_polys[:, 2, 1] = obj.vectors[support_facets][:, 0]    
                supp_polys[:, 2, 2] = obj.vectors[support_facets][:, 0] + support_vecs[:, 0]               
                supp_polys[:, 2, 3] = obj.vectors[support_facets][:, 2] + support_vecs[:, 2]       

            supp_polys = supp_polys.reshape((len(supports) * 3, 4, 3))


            supp_tris1 = np.zeros((len(supports) * 3, 3, 3))
            supp_tris1[:, 0, :] = supp_polys[:, 0, :]
            supp_tris1[:, 1, :] = supp_polys[:, 1, :]
            supp_tris1[:, 2, :] = supp_polys[:, 2, :]        

            supp_tris2 = np.zeros((len(supports) * 3, 3, 3))
            supp_tris2[:, 0, :] = supp_polys[:, 0, :]
            supp_tris2[:, 1, :] = supp_polys[:, 2, :]
            supp_tris2[:, 2, :] = supp_polys[:, 3, :]

            all_supp = np.concatenate((supp_tris1, supp_tris2))


            cols = np.zeros((all_supp.shape[0], 3))
            shades = get_poly_shades(all_supp, np.array([1.0, 1.0, 1.0]))
            cols[:, 1] = 0.7 * shades + 0.3
            supp_cols = cols

            all_polys = np.concatenate((obj.vectors, all_supp))
            all_cols = np.concatenate((obj_cols, supp_cols))

            all_poly_collection = mplot3d.art3d.Poly3DCollection(all_polys)
            all_poly_collection.set_facecolor(all_cols)
            axes.add_collection3d(all_poly_collection)

            # Auto scale to the mesh size
            scale = obj.points.flatten('K')
            axes.auto_scale_xyz(scale, scale, scale)

            self.canvas.draw()
            self.last_draw_finished = time.time()
            self.obj_waiting_to_plot = self.supports_waiting_to_plot = None


        def initUI(self):
            self.in_STL_file = tk.StringVar()
            self.in_file_entry = tk.Entry(self, textvariable=self.in_STL_file, width=80)
            self.in_file_entry.grid(row=0, column=0, columnspan=5)

            in_button = tk.Button(self, 
                                  text="Input STL File",
                                  width=16,
                                  command=self.get_in_file)
            in_button.grid(row=0, column=5)

            self.out_STL_file = tk.StringVar()
            self.out_file_label = tk.Entry(self, textvariable=self.out_STL_file, width=80)
            self.out_file_label.grid(row=1, column=0, columnspan=5)

            out_button = tk.Button(self,
                                   text="Output STL File",
                                   width=16,
                                   command=self.get_out_file)
            out_button.grid(row=1, column=5)

            lab = tk.Label(self, text="Critical angle (deg) (0 deg means all overhanging facets need support):")
            lab.grid(row=2, column=2, columnspan=3)

            self.critical_ang_txt = tk.StringVar()
            self.critical_ang_txt.set('{:.0f}'.format(DEFAULT_CRITICAL_ANGLE_DEG))
            angle_entry = tk.Entry(self, textvariable=self.critical_ang_txt)
            angle_entry.grid(row=2, column=5)

            opt_button = tk.Button(self, text="Optimize", width=16, command=self.optimize)
            opt_button.grid(row=3, column=2)

            stop_button = tk.Button(self, text="Stop", width=16, command=self.stop)
            stop_button.grid(row=3, column=3)


            self.fig = Figure(figsize=(5, 4), dpi=100)

            self.canvas = FigureCanvasTkAgg(self.fig, master=self)  # A tk.DrawingArea.
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(row=4, column=0, rowspan=3, columnspan=3)

            self.text_output = tk.Text(self, width=60, height=15)
            self.text_output.grid(row=4, column=3, rowspan=3, columnspan=3, padx=(10, 10))
            #self.text_output.insert(tk.END, 'line 1\nline 2\n')

        def get_in_file(self):
            self.in_STL_file.set(askopenfilename(filetypes =(("STL File", "*.stl"),("STL File", "*.STL")),
                                                 title = "Choose an STL file."))        
            self.load_STL()
            self.read_crit_ang()
            _, supports = get_amount_support(self.obj3d)
            self.obj_waiting_to_plot = self.obj3d
            self.supports_waiting_to_plot = supports
            self.plot_obj_support()
            #self.fig.clear()
            #t = np.arange(0, 3, .01)        
            #self.fig.add_subplot(111).plot(t, t*t)
            #self.canvas.draw()

        def read_crit_ang(self):
            try:
                ang = float(self.critical_ang_txt.get())
            except:
                ang = DEFAULT_CRITICAL_ANGLE_DEG

            self.critical_ang = ang * np.pi / 180.0
            self.text_output.insert(tk.END, 'Set critical angle to {:.0f} deg.\n'.format(ang))

        def get_out_file(self):
            self.out_STL_file.set(asksaveasfilename(filetypes =(("STL File", "*.stl"),("STL File", "*.STL")),
                                                    title = "Save Oriented STL as..."))


    app = App()
    app.grid()

    while True:
        try:
            root.mainloop()
            break
        except UnicodeDecodeError:
            pass

