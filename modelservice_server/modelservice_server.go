package main

import (
	"context"
	pb "dist_final_project/proto"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"

	"math/rand"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/peer"
)

var (
	port             = flag.Uint64("port", 8080, "server port")
	boot_ip          = flag.String("boo_ip", "localhost:8080", "boot server ip")
	key              = flag.String("key", "secret", "key for CollectModels")
	local_model_path = flag.String(
		"local_model_path",
		"my_model.pth",
		"path to local model file",
	)
	arch       = flag.String("arch", "", "Model arch")
	ip_manager = IP_Manager{Peers: make(map[string]*pb.Peer)}
)

type server struct {
	pb.UnimplementedModelServiceServer
}

type IP_Manager struct {
	mu    sync.RWMutex
	Peers map[string]*pb.Peer // should make more efficient struct??
}

func fullIP(address string, port uint32) string {
	return fmt.Sprintf("%s:%d", address, port)
}

func (ip_manager *IP_Manager) AddPeerFromContext(ctx context.Context, port uint32) error {
	p, ok := peer.FromContext(ctx)
	if !ok {
		return errors.New("could not get peer info from context")
	}
	addr, _, err := net.SplitHostPort(p.Addr.String())
	if err != nil {
		return fmt.Errorf("failed to split boot host/port: %v", err)
	}

	return ip_manager.AddPeer(addr, port)
}

func (ip_manager *IP_Manager) AddPeer(address string, port uint32) error {
	if address == "::1" {
		address = "localhost"
	}

	ip_manager.mu.Lock()
	defer ip_manager.mu.Unlock()

	ip_manager.Peers[fullIP(address, port)] = &pb.Peer{Ip: address, Port: port}

	return nil
}

// NOTE: this is creating a copy mainly for thread safety, not sure how necessary it is but better to be safe at this stage
func (ip_manager *IP_Manager) GetPeerList() map[string]*pb.Peer {
	ip_manager.mu.RLock()
	defer ip_manager.mu.RUnlock()

	peers := make(map[string]*pb.Peer, len(ip_manager.Peers))

	for k, v := range ip_manager.Peers {
		peers[k] = v
	}

	return peers
}

// TODO: needs a way of better handling updates with python client
func (s *server) GetModel(in *pb.GetModelRequest, stream pb.ModelService_GetModelServer) error {
	if err := ip_manager.AddPeerFromContext(stream.Context(), in.Port); err != nil {
		log.Printf("Warning: Failed to add peer from context: %v", err)
	}

	file, err := os.Open(*local_model_path)
	if err != nil {
		return err
	}
	defer file.Close()

	buf := make([]byte, 1024*64)
	batch_number := 0
	for {
		n, err := file.Read(buf)
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		err = stream.Send(&pb.GetModelReply{Chunk: buf[:n]})
		if err != nil {
			return err
		}
		log.Printf("Sent - batch #%v - size - %v\n", batch_number, n)
		batch_number += 1
	}

	return nil
}

func (s *server) GetBootModel(in *pb.GetBootModelRequest, stream pb.ModelService_GetBootModelServer) error {
	file, err := os.Open(fmt.Sprintf("%d_data/model_arch.py", *port))
	if err != nil {
		return err
	}
	defer file.Close()

	buf := make([]byte, 1024*64)
	batch_number := 0
	for {
		n, err := file.Read(buf)
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		err = stream.Send(&pb.GetBootModelReply{Chunk: buf[:n]})
		if err != nil {
			return err
		}
		log.Printf("Sent - batch #%v - size - %v\n", batch_number, n)
		batch_number += 1
	}

	return nil
}

func (s *server) GetPeerList(ctx context.Context, in *pb.GetPeerListRequest) (*pb.GetPeerListResponse, error) {
	err := ip_manager.AddPeerFromContext(ctx, in.Port)
	if err != nil {
		log.Printf("Error getting adding peer from context: %s", err)
	}

	log.Printf("Received: %v", in.Port)
	return &pb.GetPeerListResponse{
		Peers: ip_manager.Peers,
	}, nil
}

func getBootModel(ip string) error {

	file, err := os.Create(fmt.Sprintf("%d_data/model_arch.py", *port))
	if err != nil {
		return fmt.Errorf("failed to create file: %s", err.Error())
	}
	defer file.Close()

	conn, err := grpc.NewClient(ip, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("failed to connect: %s", err.Error())
	}
	defer conn.Close()

	client := pb.NewModelServiceClient(conn)

	stream, err := client.GetBootModel(context.Background(), &pb.GetBootModelRequest{})
	if err != nil {
		return fmt.Errorf("error getting model: %s", err.Error())
	}

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to recieve chunk: %s", err.Error())
		}

		_, err = file.Write(chunk.Chunk)
		if err != nil {
			return fmt.Errorf("failed to write chunk: %s", err.Error())
		}
	}
	log.Println("Model download complete!")

	return nil
}

func Min(a, b uint32) uint32 {
	if a < b {
		return a
	}
	return b
}

func getModel(id uint32, peer *pb.Peer) error {

	file, err := os.Create(fmt.Sprintf("%d_data/agg/model%d.pth", *port, id))
	if err != nil {
		return fmt.Errorf("failed to create file: %s", err.Error())
	}
	defer file.Close()

	conn, err := grpc.NewClient(fmt.Sprintf("%s:%d", peer.Ip, peer.Port), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("failed to connect: %s", err.Error())
	}
	defer conn.Close()

	client := pb.NewModelServiceClient(conn)

	stream, err := client.GetModel(context.Background(), &pb.GetModelRequest{Port: uint32(*port)})
	if err != nil {
		return fmt.Errorf("error getting model: %s", err.Error())
	}

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to recieve chunk: %s", err.Error())
		}

		_, err = file.Write(chunk.Chunk)
		if err != nil {
			return fmt.Errorf("failed to write chunk: %s", err.Error())
		}
	}
	log.Println("Model download complete!")

	return nil
}

func (s *server) CollectModels(_ context.Context, in *pb.CollectModelsRequest) (*pb.CollectModelsResponse, error) {

	if in.Key != *key {
		return &pb.CollectModelsResponse{
			Success: false,
		}, errors.New("unauthorized")
	}

	p := make([]*pb.Peer, 0, len(ip_manager.Peers))
	for _, v := range ip_manager.Peers {
		p = append(p, v)
		log.Printf("%s:%d", v.Ip, v.Port)
	}

	if len(ip_manager.Peers) == 0 {
		return &pb.CollectModelsResponse{
			Success: false,
		}, errors.New("No models to aggregate")
	}

	for i := uint32(1); i <= Min(10, uint32(len(p)/2)); i++ {
		chosen_peer := p[rand.Intn(len(p))]

		err := getModel(i, chosen_peer)
		if err != nil {
			return nil, err
		}
	}

	os.Create(fmt.Sprintf("%d_data/.DONE", *port))

	return &pb.CollectModelsResponse{
		Success: true,
	}, nil
}

func runServer() {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterModelServiceServer(s, &server{})

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		<-sigCh
		s.GracefulStop()
	}()

	log.Printf("server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

func copyFile(source, destination string) error {
	// Open the source file
	srcFile, err := os.Open(source)
	if err != nil {
		return fmt.Errorf("failed to open source file: %w", err)
	}
	defer srcFile.Close()

	// Create the destination file
	destFile, err := os.Create(destination)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}
	defer destFile.Close()

	// Copy the content from source to destination
	_, err = io.Copy(destFile, srcFile)
	if err != nil {
		return fmt.Errorf("failed to copy file: %w", err)
	}

	return nil
}

func boot() error {

	// Remove the directory if it exists
	// if _, err := os.Stat("%d_data"); !os.IsNotExist(err) {
	// 	fmt.Printf("Directory %s exists, removing it...\n", "%d_data")
	// 	err := os.RemoveAll("%d_data")
	// 	if err != nil {
	// 		fmt.Printf("Error removing directory: %v\n", err)
	// 		return err
	// 	}
	// }

	err := os.RemoveAll(fmt.Sprintf("%d_data", *port))
	if err != nil {
		fmt.Printf("Error removing data: %v\n", err)
		return err
	}

	err = os.Mkdir(fmt.Sprintf("%d_data", *port), 0755) // Permissions set to 0666 (read/write/execute for owner, read/execute for others)
	if err != nil {
		fmt.Printf("Error creating directory: %v\n", err)
		return err
	}

	if *arch != "" {
		err = copyFile(*arch, fmt.Sprintf("%d_data/model_arch.py", *port))
		if err != nil {
			fmt.Println("Error moving file:", err)
			return err
		}
	} else {
		err = getBootModel(*boot_ip)
		if err != nil {
			fmt.Printf("Error collecting boot model: %v\n", err)
			return err
		}

	}

	defer os.Create(fmt.Sprintf("%d_data/.BOOT_DONE", *port))

	conn, err := grpc.NewClient(*boot_ip, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()

	c := pb.NewModelServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	r, err := c.GetPeerList(ctx, &pb.GetPeerListRequest{Port: uint32(*port)})
	if err != nil {
		return err
	}

	if r.Peers == nil {
		r.Peers = make(map[string]*pb.Peer)
	}

	m := r.Peers

	ip_manager.mu.Lock()
	ip_manager.Peers = m
	ip_manager.mu.Unlock()

	host, portStr, err := net.SplitHostPort(*boot_ip)
	if err != nil {
		return fmt.Errorf("failed to split boot host/port: %v", err)
	}
	parsedPort, err := strconv.ParseUint(portStr, 10, 32)
	if err != nil {
		return fmt.Errorf("failed to parse port: %v", err)
	}
	err = ip_manager.AddPeer(host, uint32(parsedPort))
	if err != nil {
		return err
	}

	for _, v := range ip_manager.Peers {
		fmt.Printf("%s:%d\n", v.Ip, v.Port)
	}

	return nil
}

func main() {
	flag.Parse()

	if err := boot(); err != nil {
		log.Printf("Error bootsraping: %s", err.Error())
	}

	runServer()
}
