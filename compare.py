    def predict(self,
                user_prompt,
                height=1024,
                width=1024,
                seed=None,
                enhanced_prompt=None,
                negative_prompt=None,
                infer_steps=100,
                guidance_scale=6,
                batch_size=1,
                src_size_cond=(1024, 1024),
                sampler=None,
                use_style_cond=False,
                ):
        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if seed is None:
            seed = random.randint(0, 1_000_000)
        if not isinstance(seed, int):
            raise TypeError(f"`seed` must be an integer, but got {type(seed)}")
        generator = set_seeds(seed, device=self.device)
        # ========================================================================
        # Arguments: target_width, target_height
        # ========================================================================
        if width <= 0 or height <= 0:
            raise ValueError(f"`height` and `width` must be positive integers, got height={height}, width={width}")
        logger.info(f"Input (height, width) = ({height}, {width})")
        if self.infer_mode in ['fa', 'torch']:
            # We must force height and width to align to 16 and to be an integer.
            target_height = int((height // 16) * 16)
            target_width = int((width // 16) * 16)
            logger.info(f"Align to 16: (height, width) = ({target_height}, {target_width})")
        elif self.infer_mode == 'trt':
            target_width, target_height = get_standard_shape(width, height)
            logger.info(f"Align to standard shape: (height, width) = ({target_height}, {target_width})")
        else:
            raise ValueError(f"Unknown infer_mode: {self.infer_mode}")

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(user_prompt, str):
            raise TypeError(f"`user_prompt` must be a string, but got {type(user_prompt)}")
        user_prompt = user_prompt.strip()
        prompt = user_prompt

        if enhanced_prompt is not None:
            if not isinstance(enhanced_prompt, str):
                raise TypeError(f"`enhanced_prompt` must be a string, but got {type(enhanced_prompt)}")
            enhanced_prompt = enhanced_prompt.strip()
            prompt = enhanced_prompt

        # negative prompt
        if negative_prompt is None or negative_prompt == '':
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")

        # ========================================================================
        # Arguments: style. (A fixed argument. Don't Change it.)
        # ========================================================================
        if use_style_cond:
            # Only for hydit <= 1.1
            style = torch.as_tensor([0, 0] * batch_size, device=self.device)
        else:
            style = None

        # ========================================================================
        # Inner arguments: image_meta_size (Please refer to SDXL.)
        # ========================================================================
        if src_size_cond is None:
            size_cond = None
            image_meta_size = None
        else:
            # Only for hydit <= 1.1
            if isinstance(src_size_cond, int):
                src_size_cond = [src_size_cond, src_size_cond]
            if not isinstance(src_size_cond, (list, tuple)):
                raise TypeError(f"`src_size_cond` must be a list or tuple, but got {type(src_size_cond)}")
            if len(src_size_cond) != 2:
                raise ValueError(f"`src_size_cond` must be a tuple of 2 integers, but got {len(src_size_cond)}")
            size_cond = list(src_size_cond) + [target_width, target_height, 0, 0]
            image_meta_size = torch.as_tensor([size_cond] * 2 * batch_size, device=self.device)

        # ========================================================================
        start_time = time.time()
        logger.debug(f"""
                       prompt: {user_prompt}
              enhanced prompt: {enhanced_prompt}
                         seed: {seed}
              (height, width): {(target_height, target_width)}
              negative_prompt: {negative_prompt}
                   batch_size: {batch_size}
               guidance_scale: {guidance_scale}
                  infer_steps: {infer_steps}
              image_meta_size: {size_cond}
        """)
        reso = f'{target_height}x{target_width}'
        if reso in self.freqs_cis_img:
            freqs_cis_img = self.freqs_cis_img[reso]
        else:
            freqs_cis_img = self.calc_rope(target_height, target_width)

        if sampler is not None and sampler != self.sampler:
            self.pipeline, self.sampler = self.load_sampler(sampler)

        samples = self.pipeline(
            height=target_height,
            width=target_width,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=infer_steps,
            image_meta_size=image_meta_size,
            style=style,
            return_dict=False,
            generator=generator,
            freqs_cis_img=freqs_cis_img,
            use_fp16=self.args.use_fp16,
            learn_sigma=self.args.learn_sigma,
        )[0]
        gen_time = time.time() - start_time
        logger.debug(f"Success, time: {gen_time}")

        return {
            'images': samples,
            'seed': seed,
        }
