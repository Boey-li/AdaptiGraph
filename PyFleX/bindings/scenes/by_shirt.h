#pragma once
#include <iostream>
#include <vector>

inline void swap(int &a, int &b)
{
    int tmp = a;
    a = b;
    b = tmp;
}

class by_Shirt : public Scene
{
public:

    by_Shirt(const char *name) : Scene(name) {}

    float get_param_float(py::array_t<float> scene_params, int idx)
    {
        auto ptr = (float *)scene_params.request().ptr;
        float out = ptr[idx];
        return out;
    }

    void Initialize(py::array_t<float> scene_params, 
                    py::array_t<float> vertices,
                    py::array_t<int> stretch_edges,
                    py::array_t<int> bend_edges,
                    py::array_t<int> shear_edges,
                    py::array_t<int> faces,
                    int thread_idx = 0)
    {
        auto ptr = (float *)scene_params.request().ptr;
        // position 
        float initX = ptr[0];
        float initY = ptr[1];
        float initZ = ptr[2];

        // size
        int dimx = (int)ptr[3];
        int dimz = (int)ptr[4];

        // stiffness
        float stretchStiffness = ptr[5];
        float bendStiffness = ptr[6];
        float shearStiffness = ptr[7];
        int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);

        float total_mass = ptr[8];
        float radius = ptr[9];
        int render_type = ptr[10]; // 0: only points, 1: only mesh, 2: points + mesh
        int flip_mesh = int(ptr[11]); // Flip half

        // Cloth
        auto verts_buf = vertices.request();
        size_t num_verts = verts_buf.shape[0] / 3;
        if (num_verts > 0)
        {
            // If a mesh is passed, then use passed in mesh
            float mass = float(total_mass) / num_verts;
            float invMass = 1.0f / mass;
            auto lower = Vec4(initX, initY, initZ, 0);

            int baseIndex = int(g_buffers->positions.size());
            // printf("baseIndex: %d\n", baseIndex); // 0

            Vec3 velocity = Vec3(0, 0, 0);

            auto verts_ptr = (float *)verts_buf.ptr;
            for (size_t idx = 0; idx < num_verts; idx++)
            {
                g_buffers->positions.push_back(
                    Vec4(verts_ptr[3 * idx], verts_ptr[3 * idx + 1], verts_ptr[3 * idx + 2], invMass) + lower);
                g_buffers->velocities.push_back(velocity);
                g_buffers->phases.push_back(phase);
            }

            auto faces_buf = faces.request();
            auto faces_ptr = (int *)faces_buf.ptr;
            size_t num_faces = int(faces_buf.shape[0] / 3);
            for (size_t idx = 0; idx < num_faces; idx++)
            {
                g_buffers->triangles.push_back(baseIndex + faces_ptr[3 * idx]);
                g_buffers->triangles.push_back(baseIndex + faces_ptr[3 * idx + 1]);
                g_buffers->triangles.push_back(baseIndex + faces_ptr[3 * idx + 2]);
                auto p1 = g_buffers->positions[baseIndex + faces_ptr[3 * idx]];
                auto p2 = g_buffers->positions[baseIndex + faces_ptr[3 * idx + 1]];
                auto p3 = g_buffers->positions[baseIndex + faces_ptr[3 * idx + 2]];
                auto U = p2 - p1;
                auto V = p3 - p1;
                auto normal = Vec3(
                    U.y * V.z - U.z * V.y,
                    U.z * V.x - U.x * V.z,
                    U.x * V.y - U.y * V.x);
                g_buffers->triangleNormals.push_back(normal / Length(normal));
            }
            
            auto stretch_edges_buf = stretch_edges.request();
            auto stretch_edges_ptr = (int *)stretch_edges_buf.ptr;
            size_t num_stretch_edges = int(stretch_edges_buf.shape[0] / 2);
            for (size_t idx = 0; idx < num_stretch_edges; idx++)
            {
                CreateSpring(baseIndex + stretch_edges_ptr[2 * idx], baseIndex + stretch_edges_ptr[2 * idx + 1], stretchStiffness);
            }

            auto bend_edges_buf = bend_edges.request();
            auto bend_edges_ptr = (int *)bend_edges_buf.ptr;
            size_t num_bend_edges = int(bend_edges_buf.shape[0] / 2);
            for (size_t idx = 0; idx < num_bend_edges; idx++)
            {
                CreateSpring(baseIndex + bend_edges_ptr[2 * idx], baseIndex + bend_edges_ptr[2 * idx + 1], bendStiffness);
            }

            auto shear_edges_buf = shear_edges.request();
            auto shear_edges_ptr = (int *)shear_edges_buf.ptr;
            size_t num_shear_edges = int(shear_edges_buf.shape[0] / 2);
            for (size_t idx = 0; idx < num_shear_edges; idx++)
            {
                CreateSpring(baseIndex + shear_edges_ptr[2 * idx], baseIndex + shear_edges_ptr[2 * idx + 1], shearStiffness);
            }
        }
        else
        {
            float mass = float(total_mass) / (dimx * dimz); // avg bath towel is 500-700g
            CreateSpringGrid(Vec3(initX, initY, initZ), dimx, dimz, 1, radius, phase, stretchStiffness, bendStiffness, shearStiffness, 0.0f, 1.0f / mass);
        }
        
        // Flip the last half of the mesh for the folding task
        if (flip_mesh)
        {
            // int size = g_buffers->triangles.size();
            for (int j = 0; j < int((dimz - 1)); ++j)
                for (int i = int((dimx - 1) * 1 / 8); i < int((dimx - 1) * 1 / 8) + 5; ++i)
                {
                    int idx = j * (dimx - 1) + i;

                    if ((i != int((dimx - 1) * 1 / 8 + 4)))
                        swap(g_buffers->triangles[idx * 3 * 2], g_buffers->triangles[idx * 3 * 2 + 1]);
                    if ((i != int((dimx - 1) * 1 / 8)))
                        swap(g_buffers->triangles[idx * 3 * 2 + 3], g_buffers->triangles[idx * 3 * 2 + 4]);
                }
        }

        g_numSubsteps = 2; //4
        // g_params.numIterations = 30;
        g_params.numIterations = 4;

        g_params.dynamicFriction = 0.75f;
        g_params.particleFriction = 1.0f;
        g_params.damping = 1.0f;
        g_params.sleepThreshold = 0.02f;

        g_params.relaxationFactor = 1.0f;
        g_params.shapeCollisionMargin = 0.04f;

        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);
        g_drawPoints = false;

        g_params.radius = radius * 1.8f;
        g_params.collisionDistance = 0.005f;

        g_drawPoints = render_type & 1;
        g_drawCloth = (render_type & 2) >> 1;
        g_drawSprings = false;
    }

};