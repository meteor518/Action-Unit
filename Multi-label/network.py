from mxnet.gluon import nn


class RegionLayer(nn.HybridBlock):
    def __init__(self, in_channels, grid=(8, 8), layout='NCHW', **kwargs):
        super(RegionLayer, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.grid = grid
        self.layout = layout
        channel_axis = layout.find('C')

        self.region_layers = dict()

        with self.name_scope():
            for i in range(self.grid[0]):
                for j in range(self.grid[1]):
                    grid_name = 'region_conv_{}_{}'.format(i, j)
                    features = nn.HybridSequential()
                    features.add(nn.BatchNorm(axis=channel_axis))
                    features.add(nn.Activation('relu'))
                    features.add(nn.Conv2D(channels=self.in_channels, kernel_size=3,
                                           strides=1, padding=1, layout=layout))
                    self.region_layers[grid_name] = features

        for name, block in self.region_layers.items():
            self.register_child(block, name)

    def __repr__(self):
        s = '{name}({mapping}, grid={grid})'
        mapping = '{0} -> {1}'.format(self.in_channels, self.in_channels)
        return s.format(name=self.__class__.__name__, mapping=mapping, grid=self.grid)

    def hybrid_forward(self, F, x, *args, **kwargs):
        height_axis = self.layout.find('H')
        width_axis = self.layout.find('W')

        input_row_list = F.split(x, num_outputs=self.grid[0], axis=height_axis)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = F.split(row, num_outputs=self.grid[1], axis=width_axis)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                grid_name = 'region_conv_{}_{}'.format(i, j)
                grid = self.region_layers[grid_name](grid) + grid
                output_grid_list_of_a_row.append(grid)

            output_row = F.concat(*output_grid_list_of_a_row, dim=width_axis)
            output_row_list.append(output_row)

        output = F.concat(*output_row_list, dim=height_axis)
        output = F.relu(output)

        return output


class DRML(nn.HybridBlock):
    def __init__(self, classes=12, grid=(8, 8), layout='NCHW', prefix='drml', **kwargs):
        super(DRML, self).__init__(prefix=prefix, **kwargs)

        self.classes = classes
        self.grid = grid
        channel_axis = layout.find('C')

        with self.name_scope():
            self.features = nn.HybridSequential()

            # conv1
            self.features.add(nn.Conv2D(channels=32, kernel_size=11, layout=layout))

            # region2
            self.features.add(RegionLayer(in_channels=32, grid=self.grid, layout=layout))

            # pool3
            self.features.add(nn.MaxPool2D(layout=layout))
            self.features.add(nn.BatchNorm(axis=channel_axis))

            # conv4~conv7
            self.features.add(nn.Conv2D(channels=16, kernel_size=8, activation='relu', layout=layout))
            self.features.add(nn.Conv2D(channels=16, kernel_size=8, activation='relu', layout=layout))
            self.features.add(nn.Conv2D(channels=16, kernel_size=6, activation='relu', layout=layout))
            self.features.add(nn.Conv2D(channels=16, kernel_size=5, activation='relu', layout=layout))

            # fc8
            self.features.add(nn.Dense(units=4096, activation='relu'))
            self.features.add(nn.Dropout(0.5))

            # fc9
            self.features.add(nn.Dense(units=2048, activation='relu'))
            self.features.add(nn.Dropout(0.5))

            # output
            self.output = nn.Dense(units=self.classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


class PretrainedModel(nn.HybridBlock):
    def __init__(self, classes, symbol_file, param_file, ctx, **kwargs):
        super(PretrainedModel, self).__init__(prefix='')
        self.features = nn.SymbolBlock.imports(symbol_file, ['data'], param_file, ctx)
        self.output = nn.Dense(units=classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


class VGGFace(nn.HybridBlock):
    def __init__(self, classes, symbol_file, param_file, ctx, **kwargs):
        super(VGGFace, self).__init__(prefix='')
        self.features = nn.SymbolBlock.imports(symbol_file, ['data'], param_file, ctx)
        self.output = nn.Dense(units=classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


class JAA(nn.HybridBlock):
    def __init__(self, classes, symbol_file, param_file, ctx, **kwargs):
        super(JAA, self).__init__(prefix='')
        self.features = nn.SymbolBlock.imports(symbol_file, ['data'], param_file, ctx)
        self.fc1 = nn.Dense(512, activation='relu')
        self.output1 = nn.Dense(classes)
        self.fc2 = nn.Dense(512, activation='relu')
        self.output2 = nn.Dense(136)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x2 = self.fc2(x)
        x2 = self.output2(x2)
        x1 = self.fc1(x)
        x1 = F.concat(x1, x2)
        x1 = self.output1(x1)
        return F.concat(x1, x2)


class LandmarkBlock(nn.HybridBlock):
    # todo: 结构 输出激活
    def __init__(self, classes=136, prefix='landmark_'):
        super(LandmarkBlock, self).__init__(prefix)
        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(
                self.conv2d_block(80, 5),
                nn.MaxPool2D(),
                self.conv2d_block(96, 4),
                nn.MaxPool2D(),
                self.conv2d_block(128, 3),
                self.conv2d_block(128, 3),
                nn.Flatten(),
                nn.Dense(1800, activation='relu'),
                # nn.Dropout(0.5),
                nn.Dense(1000, activation='relu'),
                # nn.Dropout(0.5),
            )
            self.output = nn.Dense(classes, activation='sigmoid')  # (0,1)

    @staticmethod
    def conv2d_block(num_filters, kernel_size):
        block = nn.HybridSequential()
        block.add(nn.Conv2D(num_filters, (kernel_size, kernel_size)))
        block.add(nn.BatchNorm())
        block.add(nn.Activation(activation='relu'))
        return block

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


class MultiTaskV1(nn.HybridBlock):
    def __init__(self, classes, symbol_file, param_file, ctx, **kwargs):
        super(MultiTaskV1, self).__init__(prefix='')
        self.features = nn.SymbolBlock.imports(symbol_file, ['data'], param_file, ctx)
        self.landmark = LandmarkBlock(classes=(68 - 17 - 8) * 2)
        self.fc = nn.Dense(512, activation='relu')
        self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        landmark = self.landmark((x - 127.5) * 0.0078125)
        x = self.features(x)
        x = F.concat(x, landmark)  # 128(512)+136
        x = self.fc(x)
        x = self.output(x)
        return F.concat(x, landmark)


class MultiTaskV2(nn.HybridBlock):
    def __init__(self, classes, symbol_file, param_file, ctx, **kwargs):
        super(MultiTaskV2, self).__init__(prefix='')
        self.emb_feats = nn.SymbolBlock.imports(symbol_file, ['data'], param_file, ctx)
        self.landmark_fc1 = nn.Dense(256, activation='relu')
        self.landmark_fc2 = nn.Dense(256, activation='relu')
        self.landmark_output = nn.Dense((68 - 17 - 8) * 2, activation='sigmoid')
        # self.au_fc = nn.Dense(512, activation='relu')
        self.au_output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        feats = self.emb_feats(x)
        landmark = self.landmark_fc1(feats)
        landmark = self.landmark_fc2(landmark)
        landmark = self.landmark_output(landmark)
        au = feats
        # au = self.au_fc(au)
        au = self.au_output(au)
        return F.concat(au, landmark)
