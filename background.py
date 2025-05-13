class RandomBackgroundManager:
    def __init__(
        self,
        dataset_names: List[str] = DATASET_PARENTS.keys(),
        weights: Optional[List[float]] = None,
        final_image_size: Tuple[int, int] = (240, 320),
        load_raw_images_every: int = 5,
        load_aug_images_every: int = 1,
        load_n_raw_images: int = 10000,
        load_n_aug_images: int = 10000,
    ):
        self.dataset_names = dataset_names
        self.load_raw_images_every = load_raw_images_every
        self.load_aug_images_every = load_aug_images_every
        self.load_n_raw_images = load_n_raw_images
        self.load_n_aug_images = load_n_aug_images

        # Check if datasets are downloaded
        for dataset_name in dataset_names:
            download_dataset(dataset_name)

        # Weights for each dataset
        if weights is None:
            self.weights = np.ones(len(dataset_names))
        else:
            self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)
        assert len(weights) == len(dataset_names)

        # Index images
        self.images = {
            dataset_name: index_images([dataset_name]) for dataset_name in dataset_names
        }
        print(
            f"Indexed {len(self.images)} datasets for a total of {sum(len(v) for v in self.images.values())} images"
        )

        # Initialize augmentation pipelines
        self._safe_compress = get_safe_compress(
            compress_size=(final_image_size[0] * 4, final_image_size[1] * 4)
        )
        self._background_aug_pipeline = get_random_background_aug_pipeline(
            final_image_size=final_image_size
        )

    def sample_n_paths_from_dataset(self, dataset_name: str, n: int) -> List[str]:
        return np.random.choice(self.images[dataset_name], n, replace=True).tolist()

    def load_image(self, path: str) -> np.ndarray:
        """Load image from path as (H, W, C) numpy array."""
        image = Image.open(path)
        image = np.array(image)

        if image.ndim == 2:  # if image is grayscale, convert to RGB
            image = np.stack([image] * 3, axis=-1)

        if image.shape[-1] == 4:  # Always remove alpha channel
            image = image[..., :3]

        image = self._safe_compress(image)

        return image

    def _cache_raw_backgrounds(self, n: int):
        print(f"Caching {n} raw backgrounds")
        # sample a dataset for each background according to the weights
        dataset_names = np.random.choice(self.dataset_names, n, p=self.weights, replace=True)

        # count how many backgrounds we need from each dataset
        dataset_counts = {name: 0 for name in dataset_names}
        for name in dataset_names:
            dataset_counts[name] += 1

        # sample background paths from each dataset
        background_paths = {
            name: self.sample_n_paths_from_dataset(name, count)
            for name, count in dataset_counts.items()
        }

        # load images
        images = []
        all_paths = sum(background_paths.values(), [])
        for path in tqdm(all_paths):
            images.append(self.load_image(path))

        # shuffle images
        np.random.shuffle(images)

        self._raw_cache = images

    def _cache_aug_backgrounds(self, n: int):
        print(f"Caching {n} augmented backgrounds")
        idxs = np.random.choice(len(self._raw_cache), n, replace=True)
        self._aug_cache = []
        for idx in tqdm(idxs):
            self._aug_cache.append(
                self._background_aug_pipeline(image=self._raw_cache[idx])["image"]
            )

    def __len__(self):
        return len(self._aug_cache)

    def __getitem__(self, idx):
        return self._aug_cache[idx]

    def per_epoch_update(self, epoch: int):
        if epoch % self.load_raw_images_every == 0:
            self._cache_raw_backgrounds(self.load_n_raw_images)
        if epoch % self.load_aug_images_every == 0:
            self._cache_aug_backgrounds(self.load_n_aug_images)

    def cache_backgrounds(self):
        self._cache_raw_backgrounds(self.load_n_raw_images)
        self._cache_aug_backgrounds(self.load_n_aug_images)


if __name__ == "__main__":
    bg_manager = RandomBackgroundManager(
        # dataset_names=["mitindoors", "dtd", "openimages", "solidcolor"],
        # weights=[0.5, 0.1, 0.3, 0.1],
        dataset_names=["mitindoors"],
        weights=[1.0],
        # load_n_aug_images=1000,
        # load_n_raw_images=1000,
    )
    bg_manager.per_epoch_update(0)

    # plot images in a 8x8 grid
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(8, 8, figsize=(16, 16))

    # remove all white space
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

    # remove whitepsace between rows and columns
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(64):
        img = bg_manager[i]
        axs[i // 8, i % 8].imshow(img)
        axs[i // 8, i % 8].axis("off")

    # Remove any remaining padding from the axes
    for ax in axs.flat:
        ax.margins(0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # Save the final image with no padding
    plt.savefig("random.png", bbox_inches="tight", pad_inches=0)
7:09
class SimImageDataset(BaseBCImageData):
    def __init__(
        self,
        dataset: Path,
        pred_horizon: int,
        action_dim: int,
        task_name: str,
        image_keys: List[str],
        img_size: tuple[int, int] = [240, 320],
        n_trajectories: Optional[int] = None,
        randomize_background: bool = False,
        randomize_cameras: bool = False,
        camera_calibration: Optional[Dict[str, Any]] = None,
    ):
        print("Loading SimImageDataset...")
        super().__init__(
            (
                [
                    dh.load(p)
                    for p in tqdm(
                        [path for path in dataset.iterdir() if path.suffix == ".dex"][
                            :n_trajectories
                        ],
                        desc="Loading trajectories",
                    )
                ]
            ),
            task_name,
            device=torch.device("cpu"),
            n_pred=pred_horizon,
            action_dim=action_dim,
        )

        self.img_size = img_size
        self.render_data = mujoco.MjData(self.model)
        self.image_keys = image_keys  # TODO: add check for all keys in calib if calib on
        mujoco.mj_forward(self.model, self.render_data)

        # Maybe much of this can be moved to the Environment class?
        global_camera_poses_in_Tw = (
            None
            if camera_calibration is None
            else transform_camera_calibration(self.model, self.render_data, camera_calibration)
        )
        self.model, self.render_data = set_global_camera_xml(
            self.model, self.assets, global_camera_poses_in_Tw
        )
        self.model, self.render_data = create_wrist_cameras_xml(self.model, self.assets)
        self.nominal_params = {
            camera_name: nominal_camera_params(self.model, camera_name)
            for camera_name in self.image_keys
        }

        self.randomize_background = randomize_background
        self.randomize_cameras = randomize_cameras

    def _get_traj_step_idx(self, global_step):
        traj_ix, step = super()._get_traj_step_idx(global_step)
        if hasattr(self, "fk_data"):
            self.fk_data = mujoco.MjData(self.model)
        return traj_ix, max(0, min(step, len(self.trajs[traj_ix].data) - 1))

    def solve_fk(self):
        """
        Compute forward kinematics for the robot.
        """
        l_frame = self.fk_data.body("l_robot/attachment")
        r_frame = self.fk_data.body("r_robot/attachment")

        l_ee = torch.from_numpy(
            mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(l_frame.xquat), translation=l_frame.xpos
            ).as_matrix()
        )
        r_ee = torch.from_numpy(
            mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(r_frame.xquat), translation=r_frame.xpos
            ).as_matrix()
        )

        # Gripper state/action in task space
        l_gripper = torch.Tensor([sum(self.fk_data.qpos[7:9])])
        r_gripper = torch.Tensor([sum(self.fk_data.qpos[16:18])])

        l_pos = l_ee[:3, -1]
        r_pos = r_ee[:3, -1]
        l_6dr, r_6dr = G.matrix_to_rotation_6d(l_ee[:3, :3]), G.matrix_to_rotation_6d(r_ee[:3, :3])

        return torch.cat(
            [
                l_pos,
                l_6dr,
                l_gripper,
                r_pos,
                r_6dr,
                r_gripper,
            ]
        )

    def frame_to_state(self, traj_idx, step_idx) -> torch.Tensor:
        self.fk_data.qpos[:] = self.trajs[traj_idx].data[step_idx].obs.mj_qpos
        self.fk_data.qvel[:] = self.trajs[traj_idx].data[step_idx].obs.mj_qvel
        mujoco.mj_kinematics(self.model, self.fk_data)
        robot_state = self.solve_fk()

        # Get the parts poses
        parts_poses = torch.cat(
            [
                torch.cat(
                    [
                        torch.from_numpy(self.fk_data.body(name).xpos),
                        G.quat_wxyz_to_xyzw(torch.from_numpy(self.fk_data.body(name).xquat)),
                    ]
                )
                for name in self.part_names
            ]
        )

        return robot_state, parts_poses

    def frame_to_action(self, traj_idx, step_idx):
        cntrl = self.trajs[traj_idx].data[step_idx].act.mj_ctrl
        self.fk_data.qpos[:7] = cntrl[:7]
        self.fk_data.qpos[9:16] = cntrl[8:15]

        mujoco.mj_kinematics(self.model, self.fk_data)

        action = self.solve_fk()

        # Add the gripper control
        action[9] = cntrl[7]
        action[19] = cntrl[15]

        return action

    def frame_to_images_obs(self, traj_idx: int, step_index: int) -> Dict[str, np.ndarray]:
        self.render_data.qpos[:] = self.trajs[traj_idx].data[step_index].obs.mj_qpos
        self.render_data.qvel[:] = self.trajs[traj_idx].data[step_index].obs.mj_qvel
        if not hasattr(self, "renderer"):
            self.renderer = mujoco.Renderer(
                self.model, width=self.img_size[1], height=self.img_size[0]
            )

        images = {}

        for key in self.image_keys:

            # NOTE: Here we might get some small performance gains by randomizing all
            # cameras in a loop first, then calling `mj_forward` once.
            if self.randomize_cameras:
                randomize_camera(self.model, key, self.nominal_params[key])

            # Must do this after changing the camera in the model to propagate the changes
            mujoco.mj_forward(self.model, self.render_data)

            camera = self.model.camera(key)
            self.renderer.update_scene(self.render_data, camera=camera.id)
            images[key] = self.renderer.render()

            if self.randomize_background:
                if not hasattr(self, "bg_manager"):
                    self.bg_manager = RandomBackgroundManager(
                        dataset_names=["mitindoors"],
                        weights=[1.0],
                        load_n_aug_images=1_000,
                        load_n_raw_images=1_000,
                    )
                    self.bg_manager.per_epoch_update(0)
                random_idx = np.random.randint(0, len(self.bg_manager))
                background_image = self.bg_manager[random_idx]
                self.renderer.enable_segmentation_rendering()
                mask = self.renderer.render()[..., :1] > 0
                self.renderer.disable_segmentation_rendering()
                images[key] = np.where(mask, images[key], background_image)

        return images

    def per_epoch_update(self, epoch: int) -> None:
        if hasattr(self, "bg_manager"):
            self.bg_manager.per_epoch_update(epoch)