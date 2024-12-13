// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.5.1
// - protoc             v3.20.3
// source: proto/modelservice.proto

package proto

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.64.0 or later.
const _ = grpc.SupportPackageIsVersion9

const (
	ModelService_GetModel_FullMethodName      = "/modelservice.ModelService/GetModel"
	ModelService_GetPeerList_FullMethodName   = "/modelservice.ModelService/GetPeerList"
	ModelService_CollectModels_FullMethodName = "/modelservice.ModelService/CollectModels"
)

// ModelServiceClient is the client API for ModelService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
//
// Definition of ModelService
type ModelServiceClient interface {
	GetModel(ctx context.Context, in *GetModelRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[GetModelReply], error)
	GetPeerList(ctx context.Context, in *GetPeerListRequest, opts ...grpc.CallOption) (*GetPeerListResponse, error)
	CollectModels(ctx context.Context, in *CollectModelsRequest, opts ...grpc.CallOption) (*CollectModelsResponse, error)
}

type modelServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewModelServiceClient(cc grpc.ClientConnInterface) ModelServiceClient {
	return &modelServiceClient{cc}
}

func (c *modelServiceClient) GetModel(ctx context.Context, in *GetModelRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[GetModelReply], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &ModelService_ServiceDesc.Streams[0], ModelService_GetModel_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[GetModelRequest, GetModelReply]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type ModelService_GetModelClient = grpc.ServerStreamingClient[GetModelReply]

func (c *modelServiceClient) GetPeerList(ctx context.Context, in *GetPeerListRequest, opts ...grpc.CallOption) (*GetPeerListResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(GetPeerListResponse)
	err := c.cc.Invoke(ctx, ModelService_GetPeerList_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServiceClient) CollectModels(ctx context.Context, in *CollectModelsRequest, opts ...grpc.CallOption) (*CollectModelsResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(CollectModelsResponse)
	err := c.cc.Invoke(ctx, ModelService_CollectModels_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ModelServiceServer is the server API for ModelService service.
// All implementations must embed UnimplementedModelServiceServer
// for forward compatibility.
//
// Definition of ModelService
type ModelServiceServer interface {
	GetModel(*GetModelRequest, grpc.ServerStreamingServer[GetModelReply]) error
	GetPeerList(context.Context, *GetPeerListRequest) (*GetPeerListResponse, error)
	CollectModels(context.Context, *CollectModelsRequest) (*CollectModelsResponse, error)
	mustEmbedUnimplementedModelServiceServer()
}

// UnimplementedModelServiceServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedModelServiceServer struct{}

func (UnimplementedModelServiceServer) GetModel(*GetModelRequest, grpc.ServerStreamingServer[GetModelReply]) error {
	return status.Errorf(codes.Unimplemented, "method GetModel not implemented")
}
func (UnimplementedModelServiceServer) GetPeerList(context.Context, *GetPeerListRequest) (*GetPeerListResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetPeerList not implemented")
}
func (UnimplementedModelServiceServer) CollectModels(context.Context, *CollectModelsRequest) (*CollectModelsResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method CollectModels not implemented")
}
func (UnimplementedModelServiceServer) mustEmbedUnimplementedModelServiceServer() {}
func (UnimplementedModelServiceServer) testEmbeddedByValue()                      {}

// UnsafeModelServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to ModelServiceServer will
// result in compilation errors.
type UnsafeModelServiceServer interface {
	mustEmbedUnimplementedModelServiceServer()
}

func RegisterModelServiceServer(s grpc.ServiceRegistrar, srv ModelServiceServer) {
	// If the following call pancis, it indicates UnimplementedModelServiceServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&ModelService_ServiceDesc, srv)
}

func _ModelService_GetModel_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(GetModelRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(ModelServiceServer).GetModel(m, &grpc.GenericServerStream[GetModelRequest, GetModelReply]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type ModelService_GetModelServer = grpc.ServerStreamingServer[GetModelReply]

func _ModelService_GetPeerList_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetPeerListRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).GetPeerList(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelService_GetPeerList_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).GetPeerList(ctx, req.(*GetPeerListRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelService_CollectModels_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CollectModelsRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).CollectModels(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: ModelService_CollectModels_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).CollectModels(ctx, req.(*CollectModelsRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// ModelService_ServiceDesc is the grpc.ServiceDesc for ModelService service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var ModelService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "modelservice.ModelService",
	HandlerType: (*ModelServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "GetPeerList",
			Handler:    _ModelService_GetPeerList_Handler,
		},
		{
			MethodName: "CollectModels",
			Handler:    _ModelService_CollectModels_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "GetModel",
			Handler:       _ModelService_GetModel_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "proto/modelservice.proto",
}
