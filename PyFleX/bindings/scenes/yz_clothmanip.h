
class yz_ClothManip: public Scene
{
public:

	yz_ClothManip(const char* name) : Scene(name) {}

	void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
	{
        auto ptr = (float *) scene_params.request().ptr;

        float offset_x = ptr[0];
        float offset_y = ptr[1];
        float offset_z = ptr[2];
        int fabric_type = ptr[3];
        int dimx = ptr[4];
        int dimy = ptr[5];
        int dimz = ptr[6];
        ctrl_pts[0] = ptr[7];
        ctrl_pts[1] = ptr[8];
        ctrl_pts[2] = ptr[9];
        ctrl_pts[3] = ptr[10];
        ctrl_pts[4] = ptr[11];
        ctrl_pts[5] = ptr[12];
        ctrl_pts[6] = ptr[13];
        ctrl_pts[7] = ptr[14];

        float stretchStiffness = ptr[15];
		float bendStiffness = ptr[16];
		float shearStiffness = ptr[17];

        float dynamicFriction = ptr[18];
        float staticFriction = ptr[19];
        float particleFriction = ptr[20];

        invMass = ptr[21];
        g_ctrl = -1;

		float windStrength = ptr[22];
		float draw_mesh = ptr[23];


        g_params.dynamicFriction = dynamicFriction;
        g_params.staticFriction = staticFriction;
        g_params.particleFriction = particleFriction;

		float radius = 0.05f;

		int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);

        // void CreateSpringGrid(Vec3 lower, int dx, int dy, int dz, float radius, int phase, float stretchStiffness, float bendStiffness, float shearStiffness, Vec3 velocity, float invMass)
        if (fabric_type == 0) {
            CreateSpringGrid(
                    Vec3(offset_x, offset_y, offset_z), dimx, dimy, 1, radius, phase,
                    stretchStiffness, bendStiffness, shearStiffness, 0.0f, invMass);
        }
        else if (fabric_type == 1) {
            CreateShirt(
                    Vec3(offset_x, offset_y, offset_z), dimx, dimy, dimz, radius, phase,
                    stretchStiffness, bendStiffness, shearStiffness, 0.0f, invMass);
        }
        else if (fabric_type == 2) {
            CreatePants(
                    Vec3(offset_x, offset_y, offset_z), dimx, dimy, dimz, radius, phase,
                    stretchStiffness, bendStiffness, shearStiffness, 0.0f, invMass);
        }

		// add tethers
		for (int i=0; i < int(g_buffers->positions.size()); ++i)
		{
			// hack to rotate cloth
			// swap(g_buffers->positions[i].y, g_buffers->positions[i].z);
			// g_buffers->positions[i].y *= -1.0f;

			g_buffers->velocities[i] = RandomUnitVector()*0.1f;
		}

		g_params.radius = radius*1.0f;
		g_params.dynamicFriction = 0.25f;
		g_params.dissipation = 0.0f;
		g_params.numIterations = 4;
		g_params.drag = 0.06f;
		g_params.relaxationFactor = 1.0f;

		g_numSubsteps = 2;

		// draw options
		if (!draw_mesh) {
			g_drawPoints = true;
			g_drawSprings = true;
			g_drawMesh = false;
		}
		else {
			g_drawPoints = false;
			g_drawSprings = false;
		}

		g_windFrequency *= 2.0f;
		g_windStrength = windStrength;
	}

	int ctrl_pts[8], g_ctrl = -1;
    float invMass = 1.0f;

	void Update(py::array_t<float> update_params)
	{
        if (g_ctrl != -1) {
		    g_buffers->positions[g_ctrl].w = invMass;
            g_ctrl = -1;
        }


	    // update wind
		const Vec3 kWindDir = Vec3(0.0f, 0.0f, -1.0f);
		Vec3 wind = g_windStrength*kWindDir;
				
		g_params.wind[0] = wind.x;
		g_params.wind[1] = wind.y;
		g_params.wind[2] = wind.z;

        // update action
        auto ptr = (float *) update_params.request().ptr;

        int ctrl_idx = ptr[0];
        float dx = ptr[1];
	    float dy = ptr[2];
	    float dz = ptr[3];

        int c = ctrl_pts[ctrl_idx];

	    g_buffers->positions[c].x += dx;
	    g_buffers->positions[c].y += dy;
	    g_buffers->positions[c].z += dz;

	    g_buffers->velocities[c].x = dx / g_dt;
        g_buffers->velocities[c].y = dy / g_dt;
        g_buffers->velocities[c].z = dz / g_dt;

		g_buffers->positions[c].w = 0.0f;
        g_ctrl = c;
	}
};

