function data = readh5data(fname)
    info = h5info(fname, '/entry');

    attr = info.Attributes;
    size_attr = size(attr);

    attrNames = {};
    attrValues = {};
    for i = 1:size_attr(1)
        attrNames{i} = attr(i).Name;
        attrValues{i} = attr(i).Value;
    end

    dset = info.Datasets;
    size_dset = size(dset);

    dsetNames = {};
    destValues = {};

    for i = 1:size_dset(1)
        dsetNames{i} = dset(i).Name;
        dsetValue{i} = h5read(fname, sprintf('/entry/%s', dset(i).Name));
    end

    Names = {attrNames{:} dsetNames{:}};
    Values = {attrValues{:} dsetValue{:}};

    data = containers.Map(Names, Values);
end
